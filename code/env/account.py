import numpy as np

class Account:
    def __init__(self,  initial_budget,
                        position_cap, 
                        initial_timestep,
                        transaction_cost,
                        slippage_factor):
        ########### fixed value ###########
        self.initial_budget = initial_budget    # 예산 (KRW)
        self.position_dict = {-1 : 'short', 0 : 'hold', 1 : 'long'}
        self.position_cap = position_cap        # 최대 계약 수 상한
        
        # 시장 정보 
        self.contract_unit = 50000              # 거래 단위가 1포인트 당 5만원 (미니 선물)

        # 증거금 비율
        self.initial_margin_rate = 0.105        # 예치증거금 10.5%
        self.maintenance_margin_rate = 0.07     # 유지증거금 7%

        # 수수료, 슬리피지
        self.transaction_cost = transaction_cost
        self.slippage_factor = slippage_factor

        ########### variable value ###########
        self.current_timestep = initial_timestep # 현재 timestep (있어야 할까)

        # 계좌 (KRW)
        self.available_balance = self.initial_budget # 가용잔고
        self.margin_deposit = 0                 # 예치증거금

        # 포지션 (체결 계약)
        self.open_interest_list = []            # 미결제약정 리스트
        self.current_name_value = 0             # 보유 계약의 명목 가치 (pt)
        self.maintenance_margin = 0             # 보유 계약에 대한 유지증거금 (KRW)

        self.current_position = 0               # 현재 포지션. + / - 부호만
        self.execution_strength = 0             # 체결 계약 수
        self.total_trades = 0                   # 전체 거래 횟수

        # 현재 포지션 관련 정보
        self.average_entry = 0                  # 평균 진입가 = 보유 계약 명목 가치 / 계약 수 (pt)

        # 손익 (계좌로 계산 가능한데 따로 있어도 괜찮을 듯)
        self.realized_pnl = 0                   # 실현 손익 (KRW)
        self.unrealized_pnl = 0                 # 미실현 손익 (KRW)
        self.prev_unrealized_pnl = 0            # 직전 스텝의 미실현 손익 (KRW)
        self.total_transaction_costs = 0        # 총 수수료

        # 마진콜은 아예 안일어나게 하기 위해서 유지증거금보다 넉넉한 기준으로 기준 이상 손실이 나면 계약을 청산하도록 하자. 계약을 청산하고 나오면 마진콜 안받음


    def step(self, action, market_pt, next_timestep, get_history=True):
        '''
        action, market point에 따라 계좌, 포지션 업데이트
        새로운 계약 추가 / 계약 청산
        평균 진입가, 유지 증거금, 미실현 손익 업데이트
        일단 단위가 너무 커지는 것 같아서 전부 포인트로 계산함
        '''
        # 직전 스텝 미실현 수익 저장
        self.prev_unrealized_pnl = self.unrealized_pnl
        if self.execution_strength != 0:
            self.ave_prev_unrealized_pnl = self.prev_unrealized_pnl / self.execution_strength

        # 순 실현 손익
        realized_net_pnl = 0
        cost = 0

        # 현재 action 정보
        position = np.sign(action)
        size = abs(action)
        position_diff = position * self.current_position

        if action != 0:
            self.total_trades += 1
            cost = self._get_cost(action, market_pt)

            # 새로운 계약 체결: 현재 보유 계약이 없는 경우 / 현재 포지션과 같은 포지션을 취하는 경우
            if (self.execution_strength == 0) or (position_diff > 0):
                self._conclude_contract(size, position, market_pt)

            # 계약 청산: 현재 포지과 반대 포지션을 취하는 경우
            elif position_diff < 0:
                realized_net_pnl = self._settle_contract(size, position, market_pt)
        
        # timestep 업데이트
        self.current_timestep = next_timestep
        
        # 정보 업데이트
        self.update_account(market_pt)

        if get_history:
            return realized_net_pnl, cost
        
    def _conclude_contract(self, size, position, market_pt):
        '''
        새로운 계약 체결 함수
        '''
        self.current_position = position    # 포지션 업데이트

        name_value = size * market_pt
        initial_margin = market_pt * size * self.initial_margin_rate * self.contract_unit

        # 계약 추가
        for _ in range(size):
            self.open_interest_list.append(market_pt)
        self.current_name_value += name_value
        self.execution_strength += size

        # 총 수수료 계산
        cost = self._get_cost(size, market_pt)
        self.total_transaction_costs += cost

        # 계좌 변동
        self.available_balance -= initial_margin + cost
        self.margin_deposit += initial_margin

    def _settle_contract(self, size, position, market_pt, get_pnl=True):
        '''
        계약 청산 함수
        현재 열려있는 계약에 대해 size만큼 청산
        만약 열려있는 계약 수보다 반대 포지션을 더 많이 체결한 경우 나머지에 대해 새로운 계약 추가
        '''
        if size >= self.execution_strength:
            remain_size = size - self.execution_strength

            # 전체 계약 청산
            net_pnl = self.settle_total_contract(market_pt)

            if remain_size > 0:
                # 남은 포지션에 대해 새로운 계약 체결
                self._conclude_contract(remain_size, position, market_pt)

        else:   # 일부 청산
            settle_contract = self.open_interest_list[:size]
            settle_value = sum(settle_contract)
            pnl = self._get_pnl(market_pt, size) * self.contract_unit
            settle_initial_margin = settle_value * self.initial_margin_rate * self.contract_unit

            # 계약 청산
            del self.open_interest_list[:size]
            self.current_name_value -= settle_value
            self.execution_strength -= size

            # 총 수수료 계산
            cost = self._get_cost(size, market_pt)
            self.total_transaction_costs += cost

            # 실현 손익
            net_pnl = pnl - cost
            self.realized_pnl += net_pnl

            # 계좌 변동
            self.available_balance += settle_initial_margin + net_pnl
            self.margin_deposit -= settle_initial_margin

        if get_pnl:
            return net_pnl

    
    def settle_total_contract(self, market_pt, get_pnl=True):
        '''
        전체 계약 청산 함수
        보유한 모든 계약을 삭제, 계좌에 손익 반영, 포지션 초기화
        '''
        # 손익, 수수료, 슬리피지
        pnl = self._get_pnl(market_pt, self.execution_strength) * self.contract_unit
        cost = self._get_cost(self.execution_strength, market_pt)
        # 순손익
        net_pnl = pnl - cost
        
        # 전체 계약 청산
        self.open_interest_list.clear()
        self.current_name_value = 0
        self.current_position = 0
        self.execution_strength = 0

        self.maintenance_margin = 0

        # 실현 손익
        self.realized_pnl += net_pnl

        # 계좌 변동
        self.available_balance += self.margin_deposit + net_pnl
        self.margin_deposit = 0
        self.total_transaction_costs += cost

        # 정보 업데이트
        self.update_account(market_pt)

        if get_pnl:
            return net_pnl

    def daily_settlement(self, close_pt):
        '''
        하루 장이 마감된 후 daily settlement 이루어짐 
        '''
        if self.execution_strength != 0:
            daily_settle = self._get_pnl(close_pt, size = self.execution_strength) * self.contract_unit
            self.available_balance += daily_settle

            # 직전 스텝 미실현 수익 저장
            self.prev_unrealized_pnl = self.unrealized_pnl

            # 미실현 손익 -> 실현 손익 전환
            self.realized_pnl += daily_settle
            self.unrealized_pnl = 0

    def update_account(self, market_pt):
        if self.execution_strength != 0:
            # 평균 진입가 계산
            self.average_entry = self.current_name_value / self.execution_strength
            # 유지증거금 계산
            self.maintenance_margin = self.current_name_value * self.maintenance_margin_rate * self.contract_unit
            # 미실현 손익 계산
            self.unrealized_pnl = self._get_pnl(market_pt, self.execution_strength) * self.contract_unit

    def _get_pnl(self, market_pt, size):
        '''
        손익 계산 함수
        내 계약 중 앞 size개의 계약에 대해 입력받은 market point에 따른 손익 계산
        '''
        return sum(market_pt - self.open_interest_list[:size]) * self.current_position

    def _calculate_transaction_cost(self, action: int, market_pt) -> float:
        """행동에 따른 거래 비용 계산"""
        if action == 0:  # 거래 없으면 비용 0
            return 0.0
        
        trade_value = abs(action) * market_pt * self.contract_unit
        cost = trade_value * self.transaction_cost
        return cost

    def _calculate_slippage(self, action: int, market_pt) -> float:
        """행동에 따른 슬리피지 비용 계산"""
        if action == 0:
            return 0.0
        
        market_impact = abs(action) * self.slippage_factor
        slippage_cost = abs(action) * market_pt * self.contract_unit * market_impact
        return slippage_cost

    def _get_cost(self, action:int, market_pt):
        """행동에 따른 거래 비용 + 슬리피지 비용 계산"""
        trade_cost = self._calculate_transaction_cost(action, market_pt)
        slippage = self._calculate_slippage(action, market_pt)
        return trade_cost + slippage

    def reset(self):
        self.current_timestep = 0

        # 계좌
        self.available_balance = self.initial_budget # 가용잔고
        self.margin_deposit = 0                 # 예치증거금

        # 포지션 (체결 계약)
        self.open_interest_list = []            # 미결제약정 리스트
        self.current_name_value = 0             # 보유 계약의 명목 가치
        self.maintenance_margin = 0             # 보유 계약에 대한 유지증거금

        self.current_position = 0               # 현재 포지션. + / - 부호만
        self.execution_strength = 0             # 체결 계약 수
        self.total_trades = 0                   # 전체 거래 횟수

        # 현재 포지션 관련 정보
        self.average_entry = 0                  # 평균 진입가 = 보유 계약 명목 가치 / 계약 수

        # 손익 (계좌로 계산 가능한데 따로 있어도 괜찮을 듯)
        self.realized_pnl = 0                   # 실현 손익
        self.unrealized_pnl = 0                 # 미실현 손익
        self.prev_unrealized_pnl = 0            # 직전 스텝의 미실현 손익
        self.total_transaction_costs = 0        # 총 수수료
        
    def __str__(self):
        """계좌 상태 출력"""
        total_equity = self.available_balance + self.unrealized_pnl
        
        return (
            f"===============================================\n"
            f"📁 1. Account Status (계좌 상태)\n"
            f"💰  Available Balance  : {self.available_balance:,.0f} KRW\n"
            f"📉  Unrealized PnL     : {self.unrealized_pnl:,.0f} KRW\n"
            f"💰  Total Equity       : {total_equity:,.0f} KRW\n"
            f"⚖️  Avg Entry Price    : {self.average_entry:.2f}\n"
            f"💼  Current Position   : {self.position_dict[self.current_position]} ({self.current_position})\n"
            f"📊  Execution Strength : {self.execution_strength}/{self.position_cap}\n"
            f"===============================================\n"
        )

