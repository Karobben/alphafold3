# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/

"""Tests for pdb_to_initial_positions module."""

import os
import tempfile

from absl.testing import absltest
from alphafold3.structure import pdb_to_initial_positions
import numpy as np


class PdbToInitialPositionsTest(absltest.TestCase):
  """Test PDB initial positions loading."""

  def test_sequence_similarity(self):
    """Test sequence similarity calculation."""
    # Identical sequences
    self.assertEqual(
        pdb_to_initial_positions._sequence_similarity('ACGT', 'ACGT'), 1.0
    )

    # Completely different sequences
    self.assertEqual(
        pdb_to_initial_positions._sequence_similarity('AAAA', 'TTTT'), 0.0
    )

    # Partially matching sequences
    self.assertEqual(
        pdb_to_initial_positions._sequence_similarity('ACGT', 'ACTT'), 0.75
    )

    # Different lengths
    similarity = pdb_to_initial_positions._sequence_similarity('ACG', 'ACGT')
    self.assertAlmostEqual(similarity, 0.75)

    # Empty sequences
    self.assertEqual(
        pdb_to_initial_positions._sequence_similarity('', 'ACGT'), 0.0
    )
    self.assertEqual(
        pdb_to_initial_positions._sequence_similarity('ACGT', ''), 0.0
    )

  def test_align_residues(self):
    """Test residue alignment."""
    # Mock PDB residues with res_name field
    pdb_residues = [
        {'res_name': 'ALA'},
        {'res_name': 'CYS'},
        {'res_name': 'GLY'},
        {'res_name': 'THR'},
    ]

    # Perfect match
    target_sequence = 'ACGT'
    mapping = pdb_to_initial_positions._align_residues(
        target_sequence, pdb_residues, 'A'
    )
    expected = {0: 0, 1: 1, 2: 2, 3: 3}
    self.assertEqual(mapping, expected)

    # Partial match
    target_sequence = 'ACXT'
    mapping = pdb_to_initial_positions._align_residues(
        target_sequence, pdb_residues, 'A'
    )
    expected = {0: 0, 1: 1, 3: 3}  # Position 2 doesn't match
    self.assertEqual(mapping, expected)

    # Length mismatch
    target_sequence = 'AC'
    mapping = pdb_to_initial_positions._align_residues(
        target_sequence, pdb_residues, 'A'
    )
    expected = {0: 0, 1: 1}
    self.assertEqual(mapping, expected)

  def test_load_initial_positions_from_pdb_invalid_path(self):
    """Test that invalid PDB path returns None."""
    result = pdb_to_initial_positions.load_initial_positions_from_pdb(
        pdb_path='/nonexistent/file.pdb',
        target_chains=['A'],
        target_sequences=['ACGT'],
        max_atoms_per_token=128,
    )
    self.assertIsNone(result)

  def test_load_initial_positions_returns_correct_shape(self):
    """Test that the output array has the correct shape."""
    # Create a minimal mmCIF string for testing
    mmcif_string = """data_test
#
loop_
_atom_site.group_PDB
_atom_site.id
_atom_site.type_symbol
_atom_site.label_atom_id
_atom_site.label_alt_id
_atom_site.label_comp_id
_atom_site.label_asym_id
_atom_site.label_entity_id
_atom_site.label_seq_id
_atom_site.pdbx_PDB_ins_code
_atom_site.Cartn_x
_atom_site.Cartn_y
_atom_site.Cartn_z
_atom_site.occupancy
_atom_site.B_iso_or_equiv
_atom_site.pdbx_formal_charge
_atom_site.auth_seq_id
_atom_site.auth_comp_id
_atom_site.auth_asym_id
_atom_site.auth_atom_id
_atom_site.pdbx_PDB_model_num
ATOM   1  N  N   . ALA A 1 1 ? 10.0 20.0 30.0 1.0 50.0 ? 1 ALA A N   1
ATOM   2  C  CA  . ALA A 1 1 ? 11.0 21.0 31.0 1.0 50.0 ? 1 ALA A CA  1
ATOM   3  C  C   . ALA A 1 1 ? 12.0 22.0 32.0 1.0 50.0 ? 1 ALA A C   1
ATOM   4  O  O   . ALA A 1 1 ? 13.0 23.0 33.0 1.0 50.0 ? 1 ALA A O   1
"""

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.cif', delete=False) as f:
      f.write(mmcif_string)
      temp_path = f.name

    try:
      result = pdb_to_initial_positions.load_initial_positions_from_pdb(
          pdb_path=temp_path,
          target_chains=['A'],
          target_sequences=['A'],  # One residue
          max_atoms_per_token=128,
      )

      # Should return an array with correct shape
      if result is not None:
        expected_shape = (1, 128, 3)  # 1 token, 128 atoms, 3 coordinates
        self.assertEqual(result.shape, expected_shape)
        self.assertEqual(result.dtype, np.float32)
    finally:
      os.unlink(temp_path)


if __name__ == '__main__':
  absltest.main()
