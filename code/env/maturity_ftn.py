def calculate_maturity(dates):
    '''
    input: dates (전체 날짜, datetime 형식)
    입력 받은 모든 dates 중에서 만기일을 계산해 list로 반환하는 함수
    만기일: 매월 두 번째 목요일
    '''
    month = 0   # 월 마다 한 번이므로 월을 추적하기 위한 것
    maturity_list = []

    for date in dates:
        if date.month != month:
            if date.weekday() <= 3: # 새 월이 시작되면 계산 시작 (1주차: 첫 번째 목요일이 있는 주)
                year = date.year
                month = date.month
                yearweek = date.isocalendar().week  # 주차
                check_week = dates[(dates.year == year) & (dates.isocalendar().week == yearweek + 1)] # 월의 2주차 날짜 모두 가져오기
                
                # 1주차를 확인해야하는 경우의 예외 처리
                if len(check_week) <= 1:
                    # 조건: 2주차에 장이 열리지 않음 / 금요일만 장이 열림
                    if (len(check_week) == 0) or check_week[0].isocalender().week == 4: # 0인 경우에 뒷 조건 확인 하면 에러날텐데 or이라 상관 없는지?
                        check_week = dates[(dates.year == year) & (dates.isocalendar().week == yearweek)]

                for date in reversed(check_week): # 해당 주차의 날짜를 거꾸로 확인
                    if date.weekday() <= 3: # 목요일이 만기일 / 목요일이 없는 경우 목요일 이전 가장 가까운 날짜가 만기일
                        maturity = date.date()  # datetime.date(yyyy, mm, dd)
                        maturity_list.append(maturity)
                        break
    
    return maturity_list