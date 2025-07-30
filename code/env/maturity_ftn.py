def calculate_maturity(dates):
    '''
    input: dates (전체 날짜, datetime 형식)
    입력 받은 모든 dates 중에서 만기일을 계산해 list로 반환하는 함수
    만기일: 매월 두 번째 목요일
    '''
    month = 0   # 월 추적 변수
    maturity_list = []

    for date in dates:
        cut = date.day // 7
        weekday = date.weekday()

        # 시작 날짜가 해당 월의 만기일 이후인 경우 스킵
        if ((cut == 1) & (weekday > 3)) or ((cut == 2) & (weekday > 0)):
            month = date.month
            continue
        
        # 월이 바뀌면 실행
        if date.month == month:
            continue

        else:
            # 해당 날짜가 만기일인 경우 바로 저장 후 종료
            if (cut == 2) & (weekday == 3):
                year = date.year
                month = date.month

                maturity = date.date()  # datetime.date(yyyy, mm, dd)
                maturity_list.append(maturity)
                break

            if (weekday <= 3) or ((weekday == 4) & (cut == 0)): # 새 월이 시작되면 계산 시작 (1주차: 첫 번째 목요일이 있는 주)
                year = date.year
                month = date.month
                yearweek = date.isocalendar().week  # 기준 주차
                
                # 1주차에 장이 열리지 않은 경우의 예외 처리
                if (date.day >= 5) & (weekday <= 3):
                    check_week = dates[(dates.year == year) & (dates.isocalendar().week == yearweek)]   # 월의 2주차
                
                # 1주차에 장이 열린 경우
                else:
                    check_week = dates[(dates.year == year) & (dates.isocalendar().week == yearweek + 1)] # 월의 2주차
                
                    # 1주차를 확인해야하는 경우의 예외 처리
                    if len(check_week) <= 1:
                        # 조건: 2주차에 장이 열리지 않음 / 금요일만 장이 열림
                        if (len(check_week) == 0) or (check_week[0].isocalender().week == 4):
                            check_week = dates[(dates.year == year) & (dates.isocalendar().week == yearweek)]   # 월의 1주차

                for d in reversed(check_week): # 해당 주차의 날짜를 거꾸로 확인
                    if d.weekday() <= 3: # 목요일이 만기일 / 목요일이 없는 경우 목요일 이전 가장 가까운 날짜가 만기일
                        maturity = d.date()  # datetime.date(yyyy, mm, dd)
                        maturity_list.append(maturity)
                        break
    
    return maturity_list