import streamlit as st

import preprocessor

st.title("Chat Forensic")

st.write("Click on arrow in top left to upload your chats")

st.sidebar.title("Chat Forensic")
uploaded_file = st.sidebar.file_uploader("Choose a file")

if uploaded_file is not None:
    bytes_data = uploaded_file.getvalue()
    data = bytes_data.decode("utf-8")
    df = preprocessor.preprocess(data)

    st.dataframe(df)
    st.write(df.head(10))

    st.title("1. OVERALL FREQUENCY OF THE GROUP")

    df1 = df.copy()
    df1['message_count'] = 1

    if 'year' in df1.columns:
        df1.drop(columns='year', inplace=True)

    df1 = df1.groupby('only_date')['message_count'].sum().reset_index()
    st.write(df1)

    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.set_style("darkgrid")

    # Adjust font size and figure size
    plt.rcParams['font.size'] = 20
    plt.rcParams['figure.figsize'] = (27, 6)

    # Plotting messages sent per day over a time period
    fig, ax = plt.subplots()
    ax.plot(df1['only_date'], df1['message_count'])
    ax.set_title('Messages sent per day over a time period')

    # Display the plot in Streamlit
    st.pyplot(fig)

    st.title("2. CALCULATE TOP 10 MOST ACTIVE DAYS")

    top10days = df1.sort_values(by="message_count", ascending=False).head(10)
    top10days.reset_index(drop=True, inplace=True)
    st.write(top10days)

    # Load your dataset
    # Assuming you have a DataFrame named 'top10days'

    sns.set_style("darkgrid")

    # Adjust font size and figure size
    plt.rcParams['font.size'] = 10
    plt.rcParams['figure.figsize'] = (12, 8)


    # Function to create the bar plot
    def plot_bar_chart(data):
        fig, ax = plt.subplots()  # Create a figure and axis object
        sns.barplot(x='only_date', y='message_count', data=data, palette="hls", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)  # Rotate x-axis labels for better readability
        ax.set_title("Top 10 Days")
        ax.set_xlabel("Date")
        ax.set_ylabel("Message Count")
        plt.tight_layout()
        return fig  # Return the figure object


    # Streamlit app
    def main():

        # Sidebar to adjust parameters if needed
        # You can add options to adjust the plot here

        # Display the bar plot
        fig = plot_bar_chart(top10days)
        st.pyplot(fig)


    # Run the Streamlit app
    if __name__ == "__main__":
        main()

        total_users = len(df['user'].unique())
        total_ghosts = 237 - total_users + 1  # Assuming 237 is the total number of group members including one admin

        import streamlit as st
        import pandas as pd

        # Create a DataFrame to hold the information
        tm = {
            'Types of People': ['Total number of people who have sent at least one message in the group',
                            'Number of people who haven\'t sent even a single message in the group'],
            'Count': [total_users - 1, total_ghosts]
        }
        daf = pd.DataFrame(tm)

        # Display the DataFrame as a table
        st.table(daf)

        st.title("3. TOP 10 ACTIVE USERS")

        # Make a copy of the DataFrame
        df2 = df.copy()

        # Remove messages from "group_notification"
        df2 = df2[df2['user'] != "group_notification"]

        # Group by user and count the number of messages, then sort in descending order
        top10df = df2.groupby("user")["message"].count().sort_values(ascending=False)

        # Select the top 10 active users and reset the index
        top10df = top10df.head(10).reset_index()

        # Display the DataFrame of top 10 active users
        st.dataframe(top10df)

        # Adjust font size, figure size, and figure face color for better readability
        plt.rcParams['font.size'] = 14
        plt.rcParams['figure.figsize'] = (9, 5)
        plt.rcParams['figure.facecolor'] = '#00000000'

        import seaborn as sns
        import matplotlib.pyplot as plt
        import streamlit as st

        # Improving default styles using Seaborn
        sns.set_style("whitegrid")

        # Increasing the figure size
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plotting the line graph
        ax.plot(top10df['user'], top10df['message'], 'o--c')

        # Tilt x-axis labels at an angle
        plt.xticks(rotation=45, ha='right')

        # Labels and Title
        ax.set_xlabel('Users')
        ax.set_ylabel('Total number of messages')
        ax.set_title("Number of messages sent by group members")
        ax.legend(['Messages'])

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Improving default styles using Seaborn
        sns.set_style("whitegrid")

        # Setting the figure size
        fig, ax = plt.subplots(figsize=(12, 6))

        # Plotting the bar chart
        bar_plot = ax.bar(top10df['user'], top10df['message'], color='lightblue', label='Messages')

        # Plotting the line graph
        line_plot = ax.plot(top10df['user'], top10df['message'], 'o--c', label='Messages Trend')

        # Tilt x-axis labels at an angle
        plt.xticks(rotation=45, ha='right')

        # Labels and Title
        ax.set_xlabel('Users')
        ax.set_ylabel('Total number of messages')
        ax.set_title("Number of messages sent by group members")
        ax.legend()

        # Display the plot in Streamlit
        st.pyplot(fig)

        # Beautifying default styles using Seaborn
        sns.set_style("darkgrid")

        # Adjust font size and figure size
        plt.rcParams['font.size'] = 10
        plt.rcParams['figure.figsize'] = (12, 8)

        # Create a bar plot for top 10 active users
        fig, ax = plt.subplots()
        sns.barplot(x='user', y='message', data=top10df, palette="hls", ax=ax)

        # Tilt x-axis labels at an angle
        plt.xticks(rotation=45, ha='right')

        # Save the plot
        plt.savefig('top10_days.svg', format='svg')

        # Display the plot in Streamlit
        st.pyplot(fig)

        st.title("4. AVERAGE MESSAGE LENGTH OF TOP 10 MOST ACTIVE USERS OF THE GROUP")

        import streamlit as st

        # Calculate the message length for each message
        df2['message_length'] = df2['message'].apply(len)

        # Calculate the average message length for each user
        avg_msg_lengths = df2.groupby('user')['message_length'].mean().reset_index()
        avg_msg_lengths.rename(columns={'message_length': 'avg_message_length'}, inplace=True)

        # Merge the average message lengths with the top 10 active users
        top10df = top10df.merge(avg_msg_lengths, on='user')

        # Sort the DataFrame by average message length in descending order
        top10df_msg = top10df.sort_values(by='avg_message_length', ascending=False)

        # Display the DataFrame with average message lengths in Streamlit
        st.dataframe(top10df_msg)


        def get_colors_of_certain_order(order_list):
            """
            Generates a list of colors based on the order of unique items in order_list.

            Args:
            - order_list (list): A list of unique items

            Returns:
            - colors (list): A list of colors corresponding to the order of items in order_list
            """
            color_palette = sns.color_palette()
            num_colors = len(order_list)
            return [color_palette[i % len(color_palette)] for i in range(num_colors)]


        import seaborn as sns
        import matplotlib.pyplot as plt
        import streamlit as st

        # Set the default style using Seaborn
        sns.set_style("darkgrid")

        # Plotting multiple charts in a grid
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        # Plot 1 - Countplot of total messages sent
        bar_plot_1 = sns.barplot(x=top10df['user'], y=top10df['message'], ax=axes[0],
                                 palette=get_colors_of_certain_order(top10df['user']))

        axes[0].set_title('Total Messages Sent')
        axes[0].set_xlabel('User')
        axes[0].set_ylabel('Number of Messages Sent')

        # Rotate x-axis labels in the first plot
        axes[0].tick_params(axis='x', rotation=45)

        # Plot 2 - Barplot of the top 10 users' average message lengths
        bar_plot_2 = sns.barplot(x=top10df_msg['user'], y=top10df_msg['avg_message_length'], ax=axes[1],
                                 palette=get_colors_of_certain_order(top10df_msg['user']))

        axes[1].set_title('Average Message Lengths')
        axes[1].set_xlabel('User')
        axes[1].set_ylabel('Average Message Length')

        # Rotate x-axis labels in the second plot
        axes[1].tick_params(axis='x', rotation=45)

        # Adjust the layout to prevent overlap
        plt.tight_layout()

        # Saving the plots
        plt.savefig('top10_msg_plots_diff.svg', format='svg')

        # Display the plots in Streamlit
        st.pyplot(fig)

        st.title("5. TOP 10 USER MOST SENT MEDIA")

        import seaborn as sns
        import matplotlib.pyplot as plt

        # Ensure consistent capitalization and strip leading/trailing whitespaces
        df['message'] = df['message'].str.strip().str.lower()

        # Count occurrences of '<media omitted>' ignoring case sensitivity and leading/trailing whitespaces
        media_omitted_counts = df[df['message'].str.contains('<media omitted>', case=False)].groupby('user')[
            'message'].count()

        # Get the top 10 users with the highest count
        top_10_media_omitted_users = media_omitted_counts.nlargest(10).reset_index()

        # Rename the second column to 'most media sent'
        top_10_media_omitted_users.rename(columns={'message': 'most media sent'}, inplace=True)

        # Display the top 10 users with their counts on Streamlit
        st.write("Top 10 users with the most media sent:")
        st.write(top_10_media_omitted_users)

        # Create a bar plot
        plt.figure(figsize=(10, 6))
        sns.barplot(x='user', y='most media sent', data=top_10_media_omitted_users, palette='viridis')
        plt.title("Top 10 Users with the Highest Count of '<Media omitted>' Messages")
        plt.xlabel("User")
        plt.ylabel("Most Media Sent")
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        # Display the plot in Streamlit
        st.pyplot(plt)

    st.title('6. TOP 10 MOST USED EMOJI')

    from collections import Counter
    import emoji_data_python
    import re

    # Initialize a Counter to count emoji occurrences
    emoji_ctr = Counter()

    # Create a list of emoji characters by extracting from EmojiChar objects
    emojis_list = [emoji.char for emoji in emoji_data_python.emoji_data]

    # Compile a regular expression pattern to match emojis
    pattern = re.compile('|'.join(re.escape(p) for p in emojis_list))

    # Iterate over each row in the DataFrame
    for idx, row in df.iterrows():
        # Find emojis in the message using the compiled regular expression pattern
        emojis_found = pattern.findall(row["message"])
        # Increment the count of each emoji found
        for emoji_found in emojis_found:
            emoji_ctr[emoji_found] += 1

    import pandas as pd
    import emoji

    # Create a DataFrame to store information about the top 10 emojis
    top10emojis = pd.DataFrame(index=range(10), columns=["emoji", "emoji_count", "emoji_description"])

    # Iterate over the top 10 emojis and populate the DataFrame
    for i, (emoji_char, count) in enumerate(emoji_ctr.most_common(10)):
        # Demojize the emoji to get a plain text representation
        description = emoji.demojize(emoji_char)[1:-1]  # Remove the colons ':' at the beginning and end

        # Populate the DataFrame with emoji information
        top10emojis.at[i, 'emoji'] = emoji_char
        top10emojis.at[i, 'emoji_count'] = count
        top10emojis.at[i, 'emoji_description'] = description

    # Display the DataFrame
    st.write(top10emojis)

    # Iterate over the top 10 emojis and populate the DataFrame
    for i, (emoji_char, count) in enumerate(emoji_ctr.most_common(10)):
        # Demojize the emoji to get a plain text representation
        description = emoji.demojize(emoji_char)[1:-1]  # Remove the colons ':' at the beginning and end

        # Populate the DataFrame with emoji information
        top10emojis.at[i, 'emoji'] = emoji_char
        top10emojis.at[i, 'emoji_count'] = count
        top10emojis.at[i, 'emoji_description'] = description

    # Set figure size and font size
    plt.figure(figsize=(15, 6))
    import matplotlib

    matplotlib.rcParams['font.size'] = 15

    # Set seaborn style
    sns.set_style("darkgrid")

    # Plotting
    sns.barplot(x='emoji_count', y='emoji_description', data=top10emojis, palette="Paired_r")

    # Title and labels
    plt.title('Most Used Emojis')
    plt.xlabel('Emoji Count')
    plt.ylabel('Emoji Description')

    # Show plot
    st.pyplot(plt)

    st.title('7. MOST ACTIVE HOURS, MOST ACTIVE DAYS AND MOST ACTIVE MONTHS')

    st.header('Most Active Hours')

    # Create a copy of the DataFrame
    df3 = df.copy()

    # Set all values in 'message_count' column to 1
    df3['message_count'] = 1

    # Extract hour from datetime
    df3['hour'] = df3['date'].dt.hour

    # Group by hour and sum message counts
    grouped_by_time = df3.groupby('hour')['message_count'].sum().reset_index()

    # Convert hour values to strings
    grouped_by_time['hour'] = grouped_by_time['hour'].astype(str)

    # Define a color palette
    palette = sns.color_palette("hls", len(grouped_by_time))

    # Create a bar plot with the defined color palette
    sns.barplot(x='hour', y='message_count', data=grouped_by_time, palette=palette)

    # Add title and labels
    plt.title('Most Active Hours')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Message Count')

    # Save the plot
    plt.savefig('most_active_hours.svg', format='svg')

    # Display the plot using st.pyplot() with a specific figure
    fig, ax = plt.subplots()
    ax.bar(x='hour', height='message_count', data=grouped_by_time, color=palette)
    ax.set_xticks(range(len(grouped_by_time['hour'])))
    ax.set_xticklabels(grouped_by_time['hour'])
    ax.set_title('Most Active Hours')
    ax.set_xlabel('Hour of the Day')
    ax.set_ylabel('Message Count')

    # Add annotations to each bar
    for p in ax.patches:
        ax.annotate(f'{int(p.get_height())}',
                    (p.get_x() + p.get_width() / 2., p.get_height()),
                    ha='center', va='center', fontsize=10, color='black', xytext=(0, 5),
                    textcoords='offset points')

    st.pyplot(fig)

    st.header("Most Active Days and Months")

    # Extracting weekday and month from the datetime column
    df3['day'] = df3['date'].dt.day_name()
    df3['month'] = df3['date'].dt.month_name()

    # Specific order to be printed in for weekdays
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Grouping by day
    grouped_by_day = df3.groupby('day')['message_count'].sum().reindex(days).reset_index()

    # Specific order to be printed in for months
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    # Grouping by month
    grouped_by_month = df3.groupby('month')['message_count'].sum().reindex(months).reset_index()

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(24, 6))

    # Set font size for better readability
    import matplotlib

    matplotlib.rcParams['font.size'] = 20

    # Set seaborn style
    sns.set_style("darkgrid")

    # Plotting

    # PLOT 1: Messages grouped by day
    sns.barplot(x='day', y='message_count', data=grouped_by_day, order=days, ax=axs[0], palette='Pastel2_r')
    axs[0].set_title('Total messages sent grouped by day')

    # PLOT 2: Messages grouped by month
    sns.barplot(x='message_count', y='month', data=grouped_by_month, order=months, ax=axs[1], palette='Pastel1_d')
    axs[1].set_title('Total messages sent grouped by month')

    # Save the plots
    plt.savefig('days_and_month.svg', format='svg')

    # Display the plot using st.pyplot() with a specific figure
    st.pyplot(fig)

    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np

    # Assuming df3 is your DataFrame with necessary columns

    # Extracting weekday and month from the datetime column
    df3['day'] = df3['date'].dt.day_name()
    df3['month'] = df3['date'].dt.month_name()

    # Specific order to be printed in for weekdays
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    # Grouping by day
    grouped_by_day = df3.groupby('day')['message_count'].sum().reindex(days).reset_index()

    # Specific order to be printed in for months
    months = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October',
              'November', 'December']
    # Grouping by month
    grouped_by_month = df3.groupby('month')['message_count'].sum().reindex(months).reset_index()

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': 'polar'})

    # PLOT 1: Messages grouped by day (Spider Plot)
    angles_day = np.linspace(0, 2 * np.pi, len(days), endpoint=False).tolist()
    values_day = grouped_by_day['message_count'].tolist()
    max_value_day = max(values_day)
    values_day_normalized = [v / max_value_day for v in values_day]  # Normalize values
    axs[0].plot(angles_day, values_day_normalized, linestyle='-', color='saddlebrown', linewidth=2)
    axs[0].fill(angles_day, values_day_normalized, alpha=0.2, color='burlywood')  # Brown and antiquewhite colors

    axs[0].set_xticks(angles_day)
    axs[0].set_xticklabels(days, fontweight='bold', color='black')  # Bold and black days
    axs[0].tick_params(axis='x', colors='black')  # Set the color of the radial ticks
    axs[0].set_title('Total messages sent grouped by day', fontweight='bold', backgroundcolor='wheat',
                     pad=20)  # Added pad for more space after title

    # PLOT 2: Messages grouped by month (Spider Plot)
    angles_month = np.linspace(0, 2 * np.pi, len(months), endpoint=False).tolist()
    values_month = grouped_by_month['message_count'].tolist()
    max_value_month = max(values_month)
    values_month_normalized = [v / max_value_month for v in values_month]  # Normalize values
    axs[1].plot(angles_month, values_month_normalized, linestyle='-', color='saddlebrown', linewidth=2)
    axs[1].fill(angles_month, values_month_normalized, alpha=0.2, color='burlywood')  # Brown and antiquewhite colors

    axs[1].set_xticks(angles_month)
    axs[1].set_xticklabels(months, fontweight='bold', color='black')  # Bold and black months
    axs[1].tick_params(axis='x', colors='black')  # Set the color of the radial ticks
    axs[1].set_title('Total messages sent grouped by month', fontweight='bold', backgroundcolor='wheat',
                     pad=20)  # Added pad for more space after title

    # Adjust layout
    plt.tight_layout()

    # Display the plot using st.pyplot() with a specific figure
    st.pyplot(fig)

    st.title("8. Wordcloud")

    from wordcloud import WordCloud, STOPWORDS
    import streamlit as st
    import matplotlib.pyplot as plt

    # Initialize an empty string to store words
    comment_words = ''

    # Define stopwords to be avoided in the WordCloud
    stopwords = set(STOPWORDS)
    stopwords.update(
        ['group', 'link', 'invite', 'joined', 'message', 'deleted', 'yeah', 'hai', 'yes', 'okay', 'ok', 'will', 'use',
         'using', 'one', 'know', 'guy', 'group', 'media', 'omitted'])

    # Iterate through the DataFrame
    for val in df3.message.values:
        # Convert val to string
        val = str(val)
        # Split the value into tokens
        tokens = val.split()
        # Convert tokens to lowercase
        tokens = [token.lower() for token in tokens]
        # Append tokens to comment_words
        comment_words += " ".join(tokens) + " "

    # Generate WordCloud
    wordcloud = WordCloud(width=600, height=600, background_color='white', stopwords=stopwords,
                          min_font_size=8).generate(comment_words)

    # Display the WordCloud in Streamlit
    st.image(wordcloud.to_array(), caption='WordCloud from Messages')

    st.header("Wordcloud of Top 10 Active Users")

    from collections import Counter
    from wordcloud import WordCloud, STOPWORDS
    import matplotlib.pyplot as plt
    import streamlit as st

    # Define stopwords to be avoided in the WordCloud
    stopwords = set(STOPWORDS)
    stopwords.update(
        ['group', 'link', 'invite', 'joined', 'message', 'deleted', 'yeah', 'hai', 'yes', 'okay', 'ok', 'will', 'use',
         'using', 'one', 'know', 'guy', 'group', 'media', 'omitted'])

    # Initialize an empty string to store words for top users
    top_users_comment_words = ''

    # Iterate through the DataFrame for top users
    for user in top10df['user']:
        # Get all messages for the current user
        messages_for_user = df3[df3['user'] == user]['message']
        # Iterate through the messages
        for val in messages_for_user.values:
            # Convert val to string
            val = str(val)
            # Split the value into tokens
            tokens = val.split()
            # Convert tokens to lowercase
            tokens = [token.lower() for token in tokens]
            # Append tokens to top_users_comment_words
            top_users_comment_words += " ".join(tokens) + " "

    # Generate WordCloud for top users
    wordcloud_top_users = WordCloud(width=600, height=600, background_color='white', stopwords=stopwords,
                                    min_font_size=8).generate(top_users_comment_words)

    # Display the WordCloud in Streamlit
    st.image(wordcloud_top_users.to_array(), caption='WordCloud for Top Users')

    st.title("9.HEATMAP COMBINING MSG SENT BY MONTH AND DAY")

    import streamlit as st
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Pre-processing by day and month
    grouped_by_day_and_month = df3.groupby(['day', 'month']).size().reset_index(name='message_count')

    # Pivot the DataFrame to have days as columns, months as index, and message counts as values
    pivot_table = grouped_by_day_and_month.pivot(index='month', columns='day', values='message_count')

    # Reorder the rows to have months in chronological order
    pivot_table = pivot_table.reindex(index=months)

    # Specify the order of the days (Monday to Sunday)
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table[days_order]

    # Fill missing values with zeros
    pivot_table = pivot_table.fillna(0)

    # Reverse the color palette
    cmap = sns.color_palette("viridis_r", as_cmap=True)

    # Plotting the heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(pivot_table, cmap=cmap, annot=True, fmt='g', linewidths=.5)
    plt.title('Messages Sent by Month and Day')
    plt.xlabel('Day of the Week')
    plt.ylabel('Month')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Display the plot in Streamlit
    fig, ax = plt.subplots()
    ax = sns.heatmap(pivot_table, cmap=cmap, annot=True, fmt='g', linewidths=.5)
    plt.title('Messages Sent by Month and Day')
    plt.xlabel('Day of the Week')
    plt.ylabel('Month')
    plt.xticks(rotation=45)
    st.pyplot(fig)
