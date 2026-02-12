# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Utilities for loading initial positions from PDB files."""

import logging

from alphafold3.constants import residue_names
from alphafold3.structure import parsing
import numpy as np


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
    # Parse PDB file using existing AlphaFold 3 structure parsing
    with open(pdb_path, 'r') as f:
      pdb_string = f.read()
    
    # Parse as mmCIF format (PDB files are also supported)
    structure = parsing.from_mmcif(pdb_string)
    
    # Initialize output array
    num_tokens = sum(len(seq) for seq in target_sequences)
    positions = np.zeros((num_tokens, max_atoms_per_token, 3), dtype=np.float32)
    
    # Track which residues we've filled
    token_idx = 0
    
    # Process each target chain
    for target_chain_id, target_sequence in zip(target_chains, target_sequences):
      logging.info(f"Processing chain {target_chain_id} with {len(target_sequence)} residues")
      
      # Find matching chain in PDB structure
      pdb_chain = _find_matching_chain(structure, target_chain_id, target_sequence)
      
      if pdb_chain is None:
        logging.warning(
            f"Could not find matching chain for {target_chain_id} in PDB. "
            f"Using zeros for this chain."
        )
        token_idx += len(target_sequence)
        continue
      
      # Extract residues from PDB chain
      pdb_residues = [
          res for res in structure.iter_residues()
          if res['chain_id'] == pdb_chain
      ]
      
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
          atoms_list = [
              atom for atom in structure.iter_atoms()
              if atom['chain_id'] == pdb_chain and atom['res_id'] == pdb_residue['res_id']
          ]
          
          # Store atom coordinates (up to max_atoms_per_token)
          for atom_idx, atom in enumerate(atoms_list[:max_atoms_per_token]):
            positions[token_idx, atom_idx, 0] = atom['atom_x']
            positions[token_idx, atom_idx, 1] = atom['atom_y']
            positions[token_idx, atom_idx, 2] = atom['atom_z']
        
        token_idx += 1
    
    logging.info(f"Loaded initial positions from {pdb_path}")
    return positions
    
  except Exception as e:
    logging.error(f"Failed to load initial positions from {pdb_path}: {e}")
    return None


def _find_matching_chain(
    structure,
    target_chain_id: str,
    target_sequence: str,
) -> str | None:
  """Find the chain in PDB that best matches the target chain."""
  # First try exact chain ID match
  pdb_chains = list(structure.chains_table.id)
  if target_chain_id in pdb_chains:
    pdb_sequence = _get_chain_sequence(structure, target_chain_id)
    if _sequence_similarity(target_sequence, pdb_sequence) > 0.8:
      return target_chain_id
  
  # Try to find by sequence similarity
  best_match = None
  best_similarity = 0.0
  
  for pdb_chain_id in pdb_chains:
    pdb_sequence = _get_chain_sequence(structure, pdb_chain_id)
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


def _get_chain_sequence(structure, chain_id: str) -> str:
  """Extract sequence from a chain in the structure."""
  residues = [
      res for res in structure.iter_residues()
      if res['chain_id'] == chain_id
  ]
  sequence = ''
  for res in residues:
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
