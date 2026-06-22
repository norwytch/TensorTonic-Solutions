def beam_search(log_probs_fn, start_token, end_token, beam_width, max_len):
    """
    Returns: list of token IDs
    """
    active_beams = [([start_token], 0.0)]
    completed = []

    for _ in range(max_len):
        beam_pool = []
        for beam in active_beams:
            nt_lp=log_probs_fn(beam[0])
            for token in range(len(nt_lp)):
                candidate = (beam[0] + [token], beam[1] + nt_lp[token])
                beam_pool.append(candidate)
        beam_pool.sort(key=lambda b: b[1], reverse=True)
        beam_pool = beam_pool[:beam_width]
        active_beams = []
        for beam in beam_pool:
            if beam[0][-1] == end_token:
                completed.append(beam)
            else:
                active_beams.append(beam)
        if not active_beams:
            break
    best = max(completed or active_beams, key=lambda b: b[1])
    seq = best[0]
    if seq[-1] == end_token:
        seq = seq[:-1]
    return seq