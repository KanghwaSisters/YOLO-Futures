class Info:
    def __init__(self):
        self.status_priority = [
            'goal_profit', 'bankrupt', 'margin_call',
            'maturity_data', 'end_of_day', 'end_of_data', 'max_step'
        ]
        self.info_dict = {status: 0 for status in self.status_priority}

    def __call__(self):
        # 메인 
        activated = self._get_priority()
        if activated is None:
            return ''
        if 'max_step' in self and activated != 'max_step':
            return f"{activated}(m)"
        return activated

    def __contains__(self, status):
        # status in info 
        return self.info_dict.get(status, 0) != 0

    def __eq__(self, other):
        if isinstance(other, str):
            # 문자열 하나 (정확히 일치해야 함)
            return other in self.info_dict and self.info_dict[other] != 0

        elif isinstance(other, (list, set, tuple)):
            # 컬렉션이면, 활성 상태 중 하나라도 포함되면 True
            active_set = {status for status, v in self.info_dict.items() if v != 0}
            return not active_set.isdisjoint(other)

        return NotImplemented

    def __add__(self, status):
        # 새로운 상태를 +로 추가 
        if status not in self.info_dict:
            raise ValueError(f"Invalid status: {status}")
        self.info_dict[status] += 1
        return self

    def _get_priority(self):
        # 더 중요한 info를 대표로 출력 
        for status in self.status_priority:
            if self.info_dict[status] != 0:
                return status
        return None

    def _cal_activated_status(self):
        return sum(1 for v in self.info_dict.values() if v != 0)


if __name__ == '__main__':
    status_list = ['max_step', 'end_of_day']
    info = Info()
    info + 'max_step'
    info + 'goal_profit'

    print(info())                        # goal_profit(m)
    print('bankrupt' in info)            # False
    print(info._cal_activated_status())  # 2
    print(info in status_list)