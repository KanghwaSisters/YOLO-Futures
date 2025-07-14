import numpy as np

class Account:
    def __init__(self, initial_budget, initial_date):
        self.initial_budget = initial_budget    # 예산
        self.current_date = initial_date        # 현재 날짜

        # 계좌
        self.available_balance = initial_budget # 가용잔고
        self.margin_deposit = 0                 # 예치증거금

        # 포지션 (체결 계약)
        self.open_interest_list = []            # 미결제약정 리스트
        self.name_value = 0                     # 보유 계약의 명목 가치
        self.maintenance_margin = 0

        self.current_position = 0     
        self.position_dict = {-1 : 'short', 0 : 'hold', 1 : 'long'}
        self.execution_strength = 0             # 체결 계약 수

        # 시장 정보 
        self.contract_unit = 50000              # 거래 단위가 1포인트 당 5만원 (미니 선물)

        # 증거금 비율
        self.initial_margin_rate = 0.105        # 예치증거금 10.5%
        self.maintenance_margin_rate = 0.07     # 유지증거금 7%

        # 마진콜은 아예 안일어나게 하기 위해서 유지증거금보다 넉넉한 기준으로 기준 이상 손실이 나면 계약을 청산하도록 하자. 계약을 청산하고 나오면 마진콜 안받음


    def step(self, action, market_pt):
        # 그냥 다 포인트로 계산하자...
        if action > 0:
            self._long_position(action, market_pt)
        elif action < 0:
            self._short_position(action, market_pt)

        
    def _long_position(self, action, market_pt):
        name_value = action * market_pt
        initial_margin = market_pt * action * self.initial_margin_rate

        # 계약 저장
        for _ in range(action):
            self.open_interest_list.append(market_pt)
        self.name_value += name_value
        self.maintenance_margin = self.name_value * self.maintenance_margin_rate
        
        # 계좌 변동
        self.available_balance -= initial_margin
        self.margin_deposit += initial_margin


    def _short_position(self, action, market_pt):
        action = abs(action)
        settle_contract = self.open_interest_list[:action]
        settle_margin = sum(market_pt - settle_contract)
        settle_value = sum(settle_contract)
        settle_initial_margin = settle_contract * self.initial_margin_rate

        # 계약 청산
        del self.open_interest_list[:action]
        self.name_value -= settle_value
        self.maintenance_margin = self.name_value * self.maintenance_margin_rate

        # 계좌 변동
        self.available_balance += settle_initial_margin + settle_margin
        self.margin_deposit -= settle_initial_margin


    def daily_settlement(self, close_pt, next_date):
        # 당일 청산 -> 날짜 변경
        daily_settle = sum(close_pt - self.open_interest_list)
        self.available_balance += daily_settle

        self.maintenance_margin = self.name_value * self.maintenance_margin_rate # 중복이긴 한데 이게 실제로 작동하는 방식이라 일단 넣어둠
        self.current_date = next_date # 이거 해 말아 고민