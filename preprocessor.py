import pandas as pd
import re

def preprocess(data):
    pattern_with_century = '\d{1,2}/\d{1,2}/\d{4},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s'
    pattern_without_century = '\d{1,2}/\d{1,2}/\d{2},\s\d{1,2}:\d{2}\s[APap][mM]\s-\s'

    messages = re.split(pattern_with_century + '|' + pattern_without_century, data)[1:]
    dates = re.findall(pattern_with_century + '|' + pattern_without_century, data)

    df = pd.DataFrame({'user_message': messages, 'message_date': dates})

    # Convert message_date type, trying both formats
    try:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%Y, %I:%M %p - ')
    except ValueError:
        df['message_date'] = pd.to_datetime(df['message_date'], format='%d/%m/%y, %I:%M %p - ')

    df.rename(columns={'message_date': 'date'}, inplace=True)

    # Extracting user and message
    users = []
    messages = []
    for message in df['user_message']:
        entry = re.split('([\w\W]+?):\s', message)
        if entry[1:]:  # user name
            users.append(entry[1])
            messages.append(" ".join(entry[2:]))
        else:
            users.append('group_notification')
            messages.append(entry[0])

    df['user'] = users
    df['message'] = messages
    df.drop(columns=['user_message'], inplace=True)

    # Extracting date components
    df['only_date'] = df['date'].dt.date
    df['year'] = df['date'].dt.year
    df['month_num'] = df['date'].dt.month
    df['month'] = df['date'].dt.month_name()
    df['day'] = df['date'].dt.day
    df['day_name'] = df['date'].dt.day_name()
    df['hour'] = df['date'].dt.hour
    df['minute'] = df['date'].dt.minute

    period = []
    for hour in df[['day_name', 'hour']]['hour']:
        if hour == 23:
            period.append(str(hour) + "-" + str('00'))
        elif hour == 0:
            period.append(str('00') + "-" + str(hour + 1))
        else:
            period.append(str(hour) + "-" + str(hour + 1))

    df['period'] = period

    return df