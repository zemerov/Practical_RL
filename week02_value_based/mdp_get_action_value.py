
def get_action_value(mdp, state_values, state, action, gamma):
    """ Computes Q(s,a) as in formula above """
    
    res = 0
    
    for next_state, proba in mdp.get_next_states(state, action).items():
        res += proba * (
            mdp.get_reward(state, action, next_state) + 
            gamma * state_values[next_state]
        )
        

    return res
