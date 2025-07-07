def is_day_changed(**kwargs):
    # 날짜를 기준으로 구분 : 날짜가 달라지면 done = True 
    next_timestep = kwargs['next_timestep']
    current_timestep = kwargs['current_timestep']

    return current_timestep.date() != next_timestep.date()

def is_day_changed_wt_max(**kwargs):
    # 날짜를 기준으로 구분 : 날짜가 달라지면 done = True 
    next_timestep = kwargs['next_timestep']
    current_timestep = kwargs['current_timestep']

    is_day_changed = current_timestep.date() != next_timestep.date()
    is_max = (kwargs['max_strength'] < kwargs['current_strength'])
    
    return is_day_changed or is_max