def calculate_maturity(dates):
    # 만기일 한 번에 계산하는 함수
    # 만기일: 3, 6, 9, 12월 두번째 목요일
    # dates: datetime 형식
    dates = dates[dates.month % 3 == 0] # 3,6,9,12월만
    month = 1   # 월 마다 한 번이므로 추적하기 위한 것

    maturity_list = []
    for date in dates:
        if date.month != month:
            if date.weekday() <= 3: # 새 월이 시작되면 계산 시작 (첫 번째 목요일이 있는 주)
                year = date.year
                month = date.month
                yearweek = date.isocalendar().week  # 주차
                nextweek = dates[(dates.year == year) & (dates.isocalendar().week == yearweek + 1)] # 다음 주차 날짜 모두 가져오기

                for date in reversed(nextweek): # 다음 주차 날짜를 거꾸로 확인
                    if date.weekday() <= 3: # 목요일 혹은 목요일이 없는 경우 그 전에 가장 가까운 날짜가 만기일
                        maturity = date
                        maturity_list.append(maturity)
                        break
    
    return maturity_list