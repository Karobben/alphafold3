# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Utilities for loading initial positions from PDB files."""

import numpy as np
from alphafold3.constants import residue_names
import logging


def load_initial_positions_from_pdb(
    pdb_path: str,
    target_chains: list[str],
    target_sequences: list[str],
    max_atoms_per_token: int,
) -> np.ndarray | None:
  """Load initial atom positions from a PDB file with chain and residue matching.
  
  Args:
    pdb_path: Path to the PDB file.
    target_chains: List of chain IDs from the input (e.g., ['A', 'B']).
    target_sequences: List of sequences corresponding to target_chains.
    max_atoms_per_token: Maximum number of atoms per residue/token.
    
  Returns:
    Array of shape (num_tokens, max_atoms_per_token, 3) with coordinates in
    Ångströms, or None if loading fails.
  """
  try:
    # Parse PDB file
    with open(pdb_path, 'r') as f:
      pdb_lines = f.readlines()
    
    pdb_structure = _parse_pdb(pdb_lines)
    
    # Initialize output array
    num_tokens = sum(len(seq) for seq in target_sequences)
    positions = np.zeros((num_tokens, max_atoms_per_token, 3), dtype=np.float32)
    
    # Track which residues we've filled
    token_idx = 0
    
    # Process each target chain
    for target_chain_id, target_sequence in zip(target_chains, target_sequences):
      logging.info(f"Processing chain {target_chain_id} with {len(target_sequence)} residues")
      
      # Find matching chain in PDB structure
      pdb_chain = _find_matching_chain(pdb_structure, target_chain_id, target_sequence)
      
      if pdb_chain is None:
        logging.warning(
            f"Could not find matching chain for {target_chain_id} in PDB. "
            f"Using zeros for this chain."
        )
        token_idx += len(target_sequence)
        continue
      
      # Extract residues from PDB chain
      pdb_residues = pdb_structure['chains'][pdb_chain]['residues']
      
      # Match residues by sequence alignment
      residue_mapping = _align_residues(
          target_sequence, pdb_residues, target_chain_id
      )
      
      # Fill coordinates for each residue
      for seq_pos, target_resname in enumerate(target_sequence):
        pdb_residue_idx = residue_mapping.get(seq_pos)
        
        if pdb_residue_idx is not None and pdb_residue_idx < len(pdb_residues):
          pdb_residue = pdb_residues[pdb_residue_idx]
          
          # Get atoms for this residue from PDB
          atoms_list = pdb_residue['atoms']
          
          # Store atom coordinates (up to max_atoms_per_token)
          for atom_idx, atom in enumerate(atoms_list[:max_atoms_per_token]):
            positions[token_idx, atom_idx, 0] = atom['x']
            positions[token_idx, atom_idx, 1] = atom['y']
            positions[token_idx, atom_idx, 2] = atom['z']
        
        token_idx += 1
    
    logging.info(f"Loaded initial positions from {pdb_path}")
    return positions
    
  except Exception as e:
    logging.error(f"Failed to load initial positions from {pdb_path}: {e}")
    return None


def _parse_pdb(pdb_lines: list[str]) -> dict:
  """Parse PDB format into a structured dictionary.
  
  Args:
    pdb_lines: Lines from a PDB file.
    
  Returns:
    Dictionary with structure: {
      'chains': {
        'A': {
          'residues': [
            {'res_num': 1, 'res_name': 'ALA', 'atoms': [
              {'name': 'CA', 'x': 1.0, 'y': 2.0, 'z': 3.0}, ...
            ]},
            ...
          ]
        },
        ...
      }
    }
  """
  structure = {'chains': {}}
  
  for line in pdb_lines:
    if not line.startswith('ATOM') and not line.startswith('HETATM'):
      continue
    
    # Parse PDB ATOM/HETATM record
    # Format: ATOM serial name altLoc resName chainID resSeq iCode x y z occupancy tempFactor
    try:
      atom_name = line[12:16].strip()
      res_name = line[17:20].strip()
      chain_id = line[21:22].strip()
      res_num = int(line[22:26].strip())
      x = float(line[30:38].strip())
      y = float(line[38:46].strip())
      z = float(line[46:54].strip())
    except (ValueError, IndexError) as e:
      logging.warning(f"Failed to parse PDB line: {line.strip()}: {e}")
      continue
    
    # Initialize chain if needed
    if chain_id not in structure['chains']:
      structure['chains'][chain_id] = {'residues': []}
    
    chain_data = structure['chains'][chain_id]
    
    # Find or create residue
    residue = None
    for res in chain_data['residues']:
      if res['res_num'] == res_num:
        residue = res
        break
    
    if residue is None:
      residue = {
          'res_num': res_num,
          'res_name': res_name,
          'atoms': []
      }
      chain_data['residues'].append(residue)
    
    # Add atom
    atom = {
        'name': atom_name,
        'x': x,
        'y': y,
        'z': z
    }
    residue['atoms'].append(atom)
  
  # Sort residues by residue number in each chain
  for chain_id in structure['chains']:
    structure['chains'][chain_id]['residues'].sort(key=lambda r: r['res_num'])
  
  return structure


def _find_matching_chain(
    pdb_structure: dict,
    target_chain_id: str,
    target_sequence: str,
) -> str | None:
  """Find the chain in PDB that best matches the target chain."""
  # First try exact chain ID match
  if target_chain_id in pdb_structure['chains']:
    pdb_sequence = _get_chain_sequence(pdb_structure['chains'][target_chain_id])
    if _sequence_similarity(target_sequence, pdb_sequence) > 0.8:
      return target_chain_id
  
  # Try to find by sequence similarity
  best_match = None
  best_similarity = 0.0
  
  for pdb_chain_id in pdb_structure['chains']:
    pdb_sequence = _get_chain_sequence(pdb_structure['chains'][pdb_chain_id])
    similarity = _sequence_similarity(target_sequence, pdb_sequence)
    
    if similarity > best_similarity:
      best_similarity = similarity
      best_match = pdb_chain_id
  
  if best_similarity > 0.7:
    logging.info(
        f"Matched target chain {target_chain_id} to PDB chain {best_match} "
        f"(similarity: {best_similarity:.2f})"
    )
    return best_match
  
  return None


def _get_chain_sequence(chain_data: dict) -> str:
  """Extract sequence from a chain in the structure."""
  sequence = ''
  for res in chain_data['residues']:
    res_name = res['res_name']
    one_letter = residue_names.letters_three_to_one(res_name, default='X')
    sequence += one_letter
  return sequence


def _sequence_similarity(seq1: str, seq2: str) -> float:
  """Calculate simple sequence similarity."""
  if not seq1 or not seq2:
    return 0.0
  min_len = min(len(seq1), len(seq2))
  matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
  return matches / max(len(seq1), len(seq2))


def _align_residues(
    target_sequence: str,
    pdb_residues: list,
    chain_id: str,
) -> dict[int, int]:
  """Align target sequence to PDB residues."""
  pdb_sequence = ''
  for res in pdb_residues:
    one_letter = residue_names.letters_three_to_one(res['res_name'], default='X')
    pdb_sequence += one_letter
  
  logging.info(
      f"Chain {chain_id}: Target length={len(target_sequence)}, "
      f"PDB length={len(pdb_sequence)}"
  )
  
  mapping = {}
  min_len = min(len(target_sequence), len(pdb_sequence))
  
  for i in range(min_len):
    if target_sequence[i] == pdb_sequence[i]:
      mapping[i] = i
  
  match_rate = len(mapping) / len(target_sequence) if target_sequence else 0
  logging.info(f"Chain {chain_id}: Matched {len(mapping)}/{len(target_sequence)} residues ({match_rate:.1%})")
  
  return mapping
