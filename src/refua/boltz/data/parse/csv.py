import csv
from pathlib import Path

import numpy as np
from boltz.data import const
from boltz.data.types import MSA, MSADeletion, MSAResidue, MSASequence


def parse_csv(
    path: Path,
    max_seqs: int | None = None,
) -> MSA:
    """Process an A3M file.

    Parameters
    ----------
    path : Path
        The path to the a3m(.gz) file.
    max_seqs : int, optional
        The maximum number of sequences.

    Returns
    -------
    MSA
        The MSA object.

    """
    # Read file
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            msg = "Invalid CSV format, expected columns: ['sequence', 'key']"
            raise ValueError(msg)
        columns = {name.strip() for name in reader.fieldnames if name is not None}
        if columns != {"key", "sequence"}:
            msg = "Invalid CSV format, expected columns: ['sequence', 'key']"
            raise ValueError(msg)

        # Create taxonomy mapping
        visited = set()
        sequences = []
        deletions = []
        residues = []

        seq_idx = 0
        for row in reader:
            row = {name.strip(): value for name, value in row.items() if name is not None}
            line = row.get("sequence", "")
            if line is None:
                continue
            line = str(line).strip()
            if not line:
                continue

            key = row.get("key", "")
            if key is None:
                key = ""
            key = str(key).strip()

            # Get taxonomy, if annotated
            taxonomy_id = -1
            if key and key != "nan":
                try:
                    taxonomy_id = int(key)
                except (TypeError, ValueError):
                    try:
                        taxonomy_id = int(float(key))
                    except (TypeError, ValueError):
                        taxonomy_id = key

            # Skip if duplicate sequence
            str_seq = line.replace("-", "").upper()
            if str_seq not in visited:
                visited.add(str_seq)
            else:
                continue

            # Process sequence
            residue = []
            deletion = []
            count = 0
            res_idx = 0
            for c in line:
                if c != "-" and c.islower():
                    count += 1
                    continue
                token = const.prot_letter_to_token[c]
                token = const.token_ids[token]
                residue.append(token)
                if count > 0:
                    deletion.append((res_idx, count))
                    count = 0
                res_idx += 1

            res_start = len(residues)
            res_end = res_start + len(residue)

            del_start = len(deletions)
            del_end = del_start + len(deletion)

            sequences.append(
                (seq_idx, taxonomy_id, res_start, res_end, del_start, del_end)
            )
            residues.extend(residue)
            deletions.extend(deletion)

            seq_idx += 1
            if (max_seqs is not None) and (seq_idx >= max_seqs):
                break

    # Create MSA object
    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa
