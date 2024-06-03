import torch

class NewDRYLogitsProcessor:
    def __init__(self, multiplier: float, base: float, allowed_length: int, sequence_breakers: set[int], _range: int):
        self.multiplier = multiplier
        self.base = base
        self.allowed_length = allowed_length
        self.sequence_breakers = sequence_breakers
        self._range = _range

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        # Operations used for the algorithm are faster on CPU with numpy.
        if input_ids.type != 'cpu':
            input_ids = input_ids.to('cpu')
        input_ids = input_ids.numpy()

        if self._range > 0:
            input_ids = input_ids[:, -self._range:]

        for input_ids_row, scores_row in zip(input_ids, scores):
            # Raw integer must be extracted here to check for set membership.
            last_token = input_ids_row[-1]

            if last_token in self.sequence_breakers:
                continue

            # Exclude the last token as it always matches.
            match_indices = (input_ids_row[:-1] == last_token).nonzero()[0]

            # Stores the maximum matching sequence length
            # for each token immediately following the sequence in the input.
            match_lengths = {}

            for i in match_indices:
                next_token = input_ids_row[i+1]

                if next_token in self.sequence_breakers:
                    continue

                # We have already found that `last_token` matches at this index,
                # so the match is at least of length 1.
                match_length = 1

                # Extend the match backwards as far as possible.
                while True:
                    j = i - match_length
                    if j < 0:
                        # Start of input reached.
                        break

                    previous_token = input_ids_row[-(match_length+1)]
                    if input_ids_row[j] != previous_token:
                        # Start of match reached.
                        break

                    if previous_token in self.sequence_breakers:
                        # Sequence-breaking token reached.
                        break

                    match_length += 1

                if next_token in match_lengths:
                    match_lengths[next_token] = max(match_length, match_lengths[next_token])
                else:
                    match_lengths[next_token] = match_length

            # Apply penalties.
            for token, match_length in match_lengths.items():
                if match_length >= self.allowed_length:
                    penalty = self.multiplier * self.base ** (match_length - self.allowed_length)
                    scores_row[token] -= penalty

        return scores
