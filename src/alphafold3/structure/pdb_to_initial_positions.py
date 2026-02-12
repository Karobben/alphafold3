# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Utilities for loading initial positions from PDB files."""

from collections import defaultdict
import logging
from typing import Any

from alphafold3.constants import residue_names
from alphafold3.model.atom_layout import atom_layout
import numpy as np


# Similarity thresholds for chain matching
EXACT_MATCH_SIMILARITY_THRESHOLD = 0.8
FALLBACK_SIMILARITY_THRESHOLD = 0.7


def load_initial_positions_from_pdb(
    pdb_path: str,
    target_chains: list[str],
    target_sequences: list[str],
    atom_layout_obj: atom_layout.AtomLayout,
) -> np.ndarray | None:
  """Load initial atom positions from a PDB file with chain and residue matching.

  Args:
    pdb_path: Path to the PDB file.
    target_chains: List of chain IDs from the input (e.g., ['A', 'B']).
    target_sequences: List of sequences corresponding to target_chains.
    atom_layout_obj: The atom layout defining expected atoms per token.

  Returns:
    Array of shape (num_tokens, max_atoms_per_token, 3) with coordinates in
    Ångströms, or None if loading fails.
  """
  try:
    # Parse PDB file (line by line to avoid loading entire file into memory)
    pdb_lines = []
    with open(pdb_path, 'r') as f:
      for line in f:
        if line.startswith(('ATOM  ', 'HETATM')):
          pdb_lines.append(line)
    
    # Parse atoms from PDB
    pdb_structure = _parse_pdb_atoms(pdb_lines)
    
    # Initialize output array
    num_tokens = sum(len(seq) for seq in target_sequences)
    max_atoms = atom_layout_obj.atom_name.shape[1]
    positions = np.zeros((num_tokens, max_atoms, 3), dtype=np.float32)
    
    # Track which residues we've filled
    token_idx = 0
    
    # Process each target chain
    for target_chain_id, target_sequence in zip(target_chains, target_sequences):
      logging.info(f"Processing chain {target_chain_id} with {len(target_sequence)} residues")
      
      # Find matching chain in PDB structure
      pdb_chain_id = _find_matching_chain(
          pdb_structure, target_chain_id, target_sequence
      )
      
      if pdb_chain_id is None:
        logging.warning(
            f"Could not find matching chain for {target_chain_id} in PDB. "
            f"Using zeros for this chain."
        )
        token_idx += len(target_sequence)
        continue
      
      # Get residues from PDB chain
      pdb_residues = pdb_structure.get(pdb_chain_id, {})
      
      # Match residues by sequence alignment
      residue_mapping = _align_residues(
          target_sequence, pdb_residues, target_chain_id
      )
      
      # Fill coordinates for each residue
      for seq_pos, target_resname in enumerate(target_sequence):
        pdb_res_key = residue_mapping.get(seq_pos)
        
        if pdb_res_key is not None and pdb_res_key in pdb_residues:
          pdb_residue = pdb_residues[pdb_res_key]
          
          # Map to atom layout positions
          atom_order = atom_layout_obj.atom_name[token_idx]
          for atom_idx, atom_name in enumerate(atom_order):
            if atom_name and atom_name in pdb_residue:
              coords = pdb_residue[atom_name]
              positions[token_idx, atom_idx, :] = coords
        else:
          logging.debug(
              f"No PDB residue found for chain {target_chain_id} "
              f"position {seq_pos} ({target_resname})"
          )
        
        token_idx += 1
    
    logging.info(f"Loaded initial positions from {pdb_path}")
    return positions
    
  except Exception as e:
    logging.error(f"Failed to load initial positions from {pdb_path}: {e}")
    import traceback
    logging.error(traceback.format_exc())
    return None


def _parse_pdb_atoms(pdb_lines: list[str]) -> dict[str, dict[tuple[int, str], dict[str, np.ndarray]]]:
  """Parse PDB ATOM/HETATM records into a structured format.

  Args:
    pdb_lines: Lines from a PDB file.

  Returns:
    Dictionary mapping:
      chain_id -> (res_seq, res_name) -> atom_name -> coordinates (x, y, z)
  """
  structure = defaultdict(lambda: defaultdict(dict))
  
  for line in pdb_lines:
    if not line.startswith(('ATOM  ', 'HETATM')):
      continue
    
    try:
      atom_name = line[12:16].strip()
      res_name = line[17:20].strip()
      chain_id = line[21:22].strip()
      res_seq = int(line[22:26].strip())
      x = float(line[30:38].strip())
      y = float(line[38:46].strip())
      z = float(line[46:54].strip())
      
      # Use first alternate location only
      alt_loc = line[16:17].strip()
      if alt_loc and alt_loc != 'A':
        continue
      
      # Store coordinates
      res_key = (res_seq, res_name)
      structure[chain_id][res_key][atom_name] = np.array([x, y, z], dtype=np.float32)
      
    except (ValueError, IndexError) as e:
      logging.debug(f"Skipping malformed PDB line: {line.strip()}: {e}")
      continue
  
  return dict(structure)


def _find_matching_chain(
    pdb_structure: dict[str, dict[tuple[int, str], dict[str, np.ndarray]]],
    target_chain_id: str,
    target_sequence: str,
) -> str | None:
  """Find the chain in PDB that best matches the target chain.

  Args:
    pdb_structure: Parsed PDB structure.
    target_chain_id: Target chain ID from input.
    target_sequence: Target sequence.

  Returns:
    PDB chain ID that matches, or None if no good match.
  """
  # First try exact chain ID match
  if target_chain_id in pdb_structure:
    # Verify sequence similarity
    pdb_sequence = _get_chain_sequence(pdb_structure[target_chain_id])
    if _sequence_similarity(target_sequence, pdb_sequence) > EXACT_MATCH_SIMILARITY_THRESHOLD:
      return target_chain_id
  
  # Try to find by sequence similarity
  best_match = None
  best_similarity = 0.0
  
  for pdb_chain_id, pdb_residues in pdb_structure.items():
    pdb_sequence = _get_chain_sequence(pdb_residues)
    similarity = _sequence_similarity(target_sequence, pdb_sequence)
    
    if similarity > best_similarity:
      best_similarity = similarity
      best_match = pdb_chain_id
  
  if best_similarity > FALLBACK_SIMILARITY_THRESHOLD:
    logging.info(
        f"Matched target chain {target_chain_id} to PDB chain {best_match} "
        f"(similarity: {best_similarity:.2f})"
    )
    return best_match
  
  return None


def _get_chain_sequence(
    pdb_residues: dict[tuple[int, str], dict[str, np.ndarray]]
) -> str:
  """Extract sequence from a chain in the structure."""
  # Sort by residue number
  sorted_res = sorted(pdb_residues.keys(), key=lambda x: x[0])
  
  sequence = ''
  for _, res_name in sorted_res:
    # Convert 3-letter to 1-letter code
    one_letter = residue_names.letters_three_to_one(res_name, default='X')
    sequence += one_letter
  return sequence


def _sequence_similarity(seq1: str, seq2: str) -> float:
  """Calculate simple sequence similarity (fraction of matches)."""
  if not seq1 or not seq2:
    return 0.0
  
  min_len = min(len(seq1), len(seq2))
  matches = sum(1 for i in range(min_len) if seq1[i] == seq2[i])
  return matches / max(len(seq1), len(seq2))


def _align_residues(
    target_sequence: str,
    pdb_residues: dict[tuple[int, str], dict[str, np.ndarray]],
    chain_id: str,
) -> dict[int, tuple[int, str]]:
  """Align target sequence to PDB residues.

  This function performs a simple alignment that assumes sequences match from
  the start with minor offsets for insertions/deletions. For more complex
  sequence alignments, consider using dedicated alignment algorithms.

  Args:
    target_sequence: Target sequence (1-letter codes).
    pdb_residues: Dictionary of PDB residues keyed by (res_seq, res_name).
    chain_id: Chain ID for logging.

  Returns:
    Dict mapping target sequence position -> PDB residue key.
  """
  # Convert PDB residues to 1-letter sequence
  sorted_res = sorted(pdb_residues.keys(), key=lambda x: x[0])
  
  pdb_sequence = ''
  for _, res_name in sorted_res:
    one_letter = residue_names.letters_three_to_one(res_name, default='X')
    pdb_sequence += one_letter
  
  logging.info(
      f"Chain {chain_id}: Target length={len(target_sequence)}, "
      f"PDB length={len(pdb_sequence)}"
  )
  
  mapping = {}
  
  # Handle length differences
  min_len = min(len(target_sequence), len(pdb_sequence))
  
  for i in range(min_len):
    if target_sequence[i] == pdb_sequence[i]:
      mapping[i] = sorted_res[i]
    else:
      # Try to find nearby match
      for offset in [-1, 1, -2, 2]:
        pdb_idx = i + offset
        if 0 <= pdb_idx < len(pdb_sequence):
          if target_sequence[i] == pdb_sequence[pdb_idx]:
            mapping[i] = sorted_res[pdb_idx]
            break
  
  match_rate = len(mapping) / len(target_sequence) if target_sequence else 0
  logging.info(
      f"Chain {chain_id}: Matched {len(mapping)}/{len(target_sequence)} "
      f"residues ({match_rate:.1%})"
  )
  
  return mapping
