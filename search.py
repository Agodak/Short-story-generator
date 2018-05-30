import numpy
import copy

def generate_sample(tparams, f_init, f_next, ctx, options, beam_width, maxlen):
    """
    Generate a sample, using beam search
    """

    sample = []
    sample_score = []
    live_k = 1
    dead_k = 0

    hypothesis_samples = [[]] * live_k
    hypothesis_scores = numpy.zeros(live_k).astype('float32')
    hypothesis_states = []

    next_state = f_init(ctx)
    next_w = -1 * numpy.ones((1,)).astype('int64')

    for ii in xrange(maxlen):
        inps = [next_w, next_state]
        ret = f_next(*inps)
        next_p, next_w, next_state = ret[0], ret[1], ret[2]
        candidate_scores = hypothesis_scores[:,None] - numpy.log(next_p)
        candidate_flat = candidate_scores.flatten()

        voc_size = next_p.shape[1]
        for xx in range(len(candidate_flat) / voc_size):
            candidate_flat[voc_size * xx + 1] = 1e20

        ranks_flat = candidate_flat.argsort()[:(beam_width-dead_k)]

        voc_size = next_p.shape[1]
        trans_indices = ranks_flat / voc_size
        word_indices = ranks_flat % voc_size
        costs = candidate_flat[ranks_flat]

        new_hypothesis_samples = []
        new_hypothesis_scores = numpy.zeros(beam_width-dead_k).astype('float32')
        new_hypothesis_states = []

        for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
            new_hypothesis_samples.append(hypothesis_samples[ti]+[wi])
            new_hypothesis_scores[idx] = copy.copy(costs[idx])
            new_hypothesis_states.append(copy.copy(next_state[ti]))

        new_live_k = 0
        hypothesis_samples = []
        hypothesis_scores = []
        hypothesis_states = []

        for idx in xrange(len(new_hypothesis_samples)):
            if new_hypothesis_samples[idx][-1] == 0:
                sample.append(new_hypothesis_samples[idx])
                sample_score.append(new_hypothesis_scores[idx])
                dead_k += 1
            else:
                new_live_k += 1
                hypothesis_samples.append(new_hypothesis_samples[idx])
                hypothesis_scores.append(new_hypothesis_scores[idx])
                hypothesis_states.append(new_hypothesis_states[idx])
        hypothesis_scores = numpy.array(hypothesis_scores)
        live_k = new_live_k

        if new_live_k < 1:
            break
        if dead_k >= beam_width:
            break

        next_w = numpy.array([w[-1] for w in hypothesis_samples])
        next_state = numpy.array(hypothesis_states)

    if live_k > 0:
        for idx in xrange(live_k):
            sample.append(hypothesis_samples[idx])
            sample_score.append(hypothesis_scores[idx])

    return sample, sample_score


