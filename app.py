import nltk
import streamlit as st

# Download NLTK resources
try:
    nltk.download('punkt_tab', quiet=True)
    nltk.download('vader_lexicon', quiet=True)
    nltk.download('sentiwordnet', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('punkt', quiet=True)
except Exception as e:
    st.error(f"Error downloading NLTK resources: {e}")

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

    st.title("10. THREE METHODS FOR IDENTIFYING SPAM MESSAGES IN GROUP CHAT")

    import pandas as pd

    # Assuming df is your DataFrame containing the chat data
    # Assuming 'message' column contains the messages and 'user' column contains the user information

    # Filter messages with more than 3 words
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    filtered_df = df[df['word_count'] > 3]

    # Count the occurrences of each message
    message_counts = filtered_df['message'].value_counts()

    # Filter messages sent 10 or more times
    duplicates = message_counts[message_counts >= 10]

    # Get the messages sent 10 or more times
    duplicates_messages = duplicates.index.tolist()

    # Filter the DataFrame to get rows with duplicate messages
    duplicate_rows = filtered_df[filtered_df['message'].isin(duplicates_messages)]

    # Group by message and user, then count the occurrences
    duplicate_counts = duplicate_rows.groupby(['message', 'user']).size().reset_index(name='count')

    # Filter to keep only messages sent 10 or more times
    duplicate_counts = duplicate_counts[duplicate_counts['count'] >= 10]

    # Display the DataFrame
    st.write(duplicate_counts)

    st.header("Messages having more than 70 words")

    # SHOW THE MESSAGES THAT HAVE WORD COUNT OF MORE THAN 70 WORDS AND MAY BE THE SPAM MESSAGES!!
    import pandas as pd

    # Assuming df is your DataFrame containing the chat data
    # Assuming 'message' column contains the messages and 'user' column contains the user information

    # Filter messages with more than 5 words
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))
    filtered_df = df[df['word_count'] > 70]

    # Select only the 'user' and 'message' columns
    filtered_messages = filtered_df[['user', 'message']]

    # Display the DataFrame
    st.write(filtered_messages)

    # SHOW THE MESSAGES THAT HAVE WORD COUNT OF MORE THAN 15 WORDS AND SENT MORE THAN ONE TIME AND MAY BE THE SPAM MESSAGES!!
    import pandas as pd

    # Assuming df is your DataFrame containing the chat data
    # Assuming 'message' column contains the messages and 'user' column contains the user information

    # Calculate the word count for each message
    df['word_count'] = df['message'].apply(lambda x: len(x.split()))

    # Filter messages with more than 100 words
    filtered_df = df[df['word_count'] > 15]

    # Group by message and user, then count the occurrences
    message_counts = filtered_df.groupby(['message', 'user']).size().reset_index(name='count')

    # Filter to keep only messages sent more than once
    message_counts = message_counts[message_counts['count'] > 1]

    # Select only the 'user', 'message', and 'count' columns
    filtered_messages = message_counts[['user', 'message', 'count']]

    # Display the DataFrame
    st.write(filtered_messages)

    st.title("11. Extract potential keywords related to criminal activities")


    def extract_potential_criminal_keywords(text):
        # Define a list of keywords related to criminal activities
        keywords = ['murder', 'homicide', 'assault', 'battery', 'robbery', 'theft', 'burglary', 'larceny', 'arson',
                    'kidnapping', 'abduction', 'ransom', 'extortion', 'blackmail', 'carjacking', 'home invasion',
                    'sexual assault', 'rape', 'sexual abuse', 'pedophilia', 'child abuse', 'child neglect',
                    'domestic violence', 'stalking', 'harassment', 'cyberbullying', 'bullying', 'terrorism',
                    'terrorist attack', 'bombing', 'shooting', 'mass shooting', 'hijacking', 'hijack', 'hate crime',
                    'hate speech', 'ethnic cleansing', 'genocide', 'war crime', 'torture', 'human rights violation',
                    'death penalty', 'execution', 'suicide', 'suicide bombing', 'illegal arms dealing',
                    'weapons trafficking', 'gun trafficking', 'illegal firearm possession', 'organized robbery',
                    'money counterfeiting', 'document forgery', 'identity fraud', 'embezzlement scheme',
                    'ponzi investment', 'corporate bribery', 'political scandal', 'election fraud', 'voter fraud',
                    'false testimony', 'perjury', 'public corruption', 'judicial bribery', 'court tampering',
                    'legal malpractice', 'academic dishonesty', 'academic fraud', 'plagiarism', 'cyber espionage',
                    'cyber terrorism', 'cyber attack', 'malware', 'hacking', 'phishing', 'identity theft',
                    'cyber theft', 'money laundering', 'laundering operation', 'illegal money transfer', 'tax evasion',
                    'tax avoidance', 'offshore account', 'illegal wildlife trade', 'poaching',
                    'endangered species trafficking', 'environmental crime', 'pollution crime', 'ecological disaster',
                    'oil spill', 'hazardous waste dumping', 'illegal logging', 'wildlife smuggling',
                    'ivory trafficking', 'illegal fishing', 'overfishing', 'piracy', 'copyright infringement',
                    'software piracy', 'film piracy', 'music piracy', 'bootlegging', 'alcohol smuggling',
                    'drug cultivation', 'illegal mining', 'blood diamond', 'conflict diamond', 'antiquities smuggling',
                    'art theft', 'artifact smuggling', 'cultural heritage crime', 'cultural property theft',
                    'human experimentation', 'unethical research', 'medical malpractice', 'patient abuse',
                    'insurance scam', 'healthcare fraud', 'pharmaceutical fraud', 'product tampering',
                    'food adulteration', 'food contamination', 'product recall', 'workplace harassment',
                    'workplace discrimination', 'labor exploitation', 'slavery', 'forced labor', 'child labor',
                    'child soldiering', 'child marriage', 'forced marriage', 'arranged marriage',
                    'arranged child marriage', 'early marriage', 'forced prostitution', 'child prostitution',
                    'sex tourism', 'sex slavery', 'forced abortion', 'gender-based violence', 'honor killing',
                    'female genital mutilation', 'forced sterilization', 'forced organ harvesting', 'organ trafficking',
                    'illegal adoption', 'baby trafficking', 'baby farming', 'child soldier recruitment',
                    'child soldier use', 'child soldier rehabilitation', 'child soldier reintegration',
                    'child soldier demobilization', 'drug', 'drugs', 'narcotic', 'narcotics', 'trafficking',
                    'trafficker', 'traffickers', 'trafficking ring', 'drug cartel', 'drug lord', 'drug trade',
                    'drug smuggling', 'drug dealing', 'drug abuse', 'drug addiction', 'substance abuse', 'illegal drug',
                    'illicit drug', 'cocaine', 'heroin', 'marijuana', 'cannabis', 'methamphetamine', 'ecstasy', 'LSD',
                    'crack cocaine', 'opium', 'fentanyl', 'prescription drug abuse', 'prescription drug fraud',
                    'prescription drug trafficking', 'opioid trafficking', 'opioid epidemic', 'synthetic drug',
                    'designer drug', 'party drug', 'performance-enhancing drug', 'doping', 'steroid',
                    'anabolic steroid', 'steroid trafficking', 'steroid abuse', 'steroid scandal', 'fraud',
                    'fraudulent', 'fraudster', 'fraudulence', 'scam', 'scams', 'scammer', 'scamming', 'scam artist',
                    'con artist', 'confidence trick', 'confidence scheme', 'pyramid scheme', 'Ponzi scheme',
                    'investment fraud', 'securities fraud', 'wire fraud', 'mail fraud', 'identity theft',
                    'credit card fraud', 'insurance fraud', 'tax fraud', 'welfare fraud', 'mortgage fraud',
                    'bank fraud', 'accounting fraud', 'corporate fraud', 'white-collar crime', 'embezzlement',
                    'forgery', 'counterfeiting', 'money laundering', 'launder money', 'laundering scheme',
                    'dirty money', 'black money', 'illegal money', 'underground economy', 'corruption', 'corrupt',
                    'corrupt practices', 'bribery', 'bribe', 'kickback', 'graft', 'embezzle', 'embezzler',
                    'corrupt official', 'political corruption', 'police corruption', 'judicial corruption',
                    'organized crime', 'racketeering', 'mafia', 'mob', 'mobster', 'syndicate', 'criminal organization',
                    'underworld', 'illegal trade', 'black market', 'illegal gambling', 'human trafficking',
                    'sex trafficking', 'child trafficking', 'organ trafficking', 'illegal arms trade', 'arms smuggling',
                    'terrorist financing', 'cybercrime', 'cybercriminal', 'cyber fraud', 'cyber scam', 'dark web',
                    'illicit trade', 'contraband', 'contraband smuggling', 'contraband trade', 'contraband trafficking',
                    'illicit goods', 'illegal immigration', 'undocumented immigration', 'human smuggling',
                    'border smuggling', 'smuggling route', 'smuggling operation', 'smuggling network', 'scam']
        # Compile a regular expression pattern to match the keywords
        pattern = re.compile('|'.join(keywords), flags=re.IGNORECASE)
        # Find all occurrences of keywords in the text
        matches = pattern.findall(text)
        # Return a list of unique matched keywords
        return list(set(matches))


    # Apply the function to extract potential criminal keywords from each message
    df['potential_criminal_keywords'] = df['message'].apply(extract_potential_criminal_keywords)

    # Filter messages with potential criminal keywords
    potential_criminal_messages = df[df['potential_criminal_keywords'].apply(len) > 0]

    # Display the messages containing potential criminal keywords along with the usernames
    st.write(potential_criminal_messages[['user', 'message', 'potential_criminal_keywords']])

    st.title("12.Extract links from messages and return the user who sent the message")

    import streamlit as st
    import pandas as pd
    import re

    def extract_links_and_user(text):
        # Regular expression pattern to match URLs
        url_pattern = r'https?://\S+|www\.\S+'
        # Find all occurrences of URLs in the text
        urls = re.findall(url_pattern, text)
        # Return a tuple containing the user and the list of URLs found
        return urls


    # Assuming df is your DataFrame containing the chat data
    # Assuming 'message' column contains the messages and 'user' column contains the user information

    # Apply the function to extract links and users from each message
    df['Links'] = df.apply(lambda row: extract_links_and_user(row['message']), axis=1)

    # Filter messages with links
    messages_with_links = df[df['Links'].apply(lambda x: len(x)) > 0]

    # Display the messages containing links along with the usernames in a scrollable Streamlit table
    st.write(messages_with_links[['user', 'Links']])

    st.title("13.TO FIND TOP 10 MOST SHARED LINKS ALONG WITH SOME OTHER DETAILS")

    import streamlit as st
    import pandas as pd
    import re
    from collections import Counter


    # Function to extract links from messages
    def extract_links(text):
        # Regular expression pattern to match URLs
        url_pattern = r'https?://\S+|www\.\S+'
        # Find all occurrences of URLs in the text
        urls = re.findall(url_pattern, text)
        # Return the list of URLs found
        return urls


    # Assuming df is your DataFrame containing the chat data
    # Assuming 'message' column contains the messages and 'user' column contains the user information
    # Assuming 'date_time' column contains the date and time information

    # Apply the function to extract links from each message
    df['links'] = df['message'].apply(extract_links)

    # Flatten the list of links
    all_links = [link for links in df['links'] for link in links]

    # Count the frequency of each link
    link_counts = Counter(all_links)

    # Get the top 10 most shared links
    top_10_links = link_counts.most_common(10)

    # Create a dictionary to store users who shared each link along with the count of shares and date_time
    link_users_count_date = {link: Counter() for link, _ in top_10_links}

    # Iterate through each row to populate the link_users_count_date dictionary
    for index, row in df.iterrows():
        for link in row['links']:
            if link in link_users_count_date:
                link_users_count_date[link][(row['user'], row['date'])] += 1

    # Create a DataFrame to store the information
    data = []

    # Iterate through the top 10 most shared links
    for link, count in top_10_links:
        for (user, date), share_count in link_users_count_date[link].items():
            data.append([link, count, user, share_count, date])

    # Create DataFrame from data list
    link_data_df = pd.DataFrame(data, columns=['Link', 'Link Count', 'User', 'Share Count', 'Date/Time'])

    # Display the information in a scrollable table using Streamlit
    st.write(link_data_df)

    st.title("14. Sentimental Analysis")

    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import sentiwordnet as swn

    # Sentiment analysis with VADER
    def analyze_sentiment_vader(sentence):
        sia = SentimentIntensityAnalyzer()
        sentiment_scores = sia.polarity_scores(sentence)
        return sentiment_scores


    # Sentiment analysis with SentiWordNet
    def analyze_sentiment_sentiwordnet(sentence):
        sentiment_score = {'pos': 0, 'neg': 0, 'obj': 0}
        tokens = nltk.word_tokenize(sentence)
        for token in tokens:
            synsets = list(swn.senti_synsets(token))
            if synsets:
                synset = synsets[0]  # Take the first synset (most common meaning)
                sentiment_score['pos'] += synset.pos_score()
                sentiment_score['neg'] += synset.neg_score()
                sentiment_score['obj'] += synset.obj_score()
        return sentiment_score


    # Example usage:
    sample_sentence = "I love this product. It's amazing!"
    vader_sentiment = analyze_sentiment_vader(sample_sentence)
    sentiwordnet_sentiment = analyze_sentiment_sentiwordnet(sample_sentence)

    st.write("VADER Sentiment:", vader_sentiment)
    st.write("SentiWordNet Sentiment:", sentiwordnet_sentiment)

    # Function to perform sentiment analysis on each message in the DataFrame
    def analyze_sentiment_df(df):
        vader_sentiments = []
        sentiwordnet_sentiments = []
        for message in df['message']:
            vader_sentiment = analyze_sentiment_vader(message)
            sentiwordnet_sentiment = analyze_sentiment_sentiwordnet(message)
            vader_sentiments.append(vader_sentiment)
            sentiwordnet_sentiments.append(sentiwordnet_sentiment)
        return vader_sentiments, sentiwordnet_sentiments


    # Example usage:
    # Assuming df is your DataFrame containing the messages
    vader_sentiments, sentiwordnet_sentiments = analyze_sentiment_df(df)

    # Adding sentiment scores to the DataFrame
    df['vader_sentiment'] = vader_sentiments
    df['sentiwordnet_sentiment'] = sentiwordnet_sentiments

    # Displaying the DataFrame with sentiment scores
    st.write(df[['message', 'vader_sentiment', 'sentiwordnet_sentiment']])

    import streamlit as st
    import matplotlib.pyplot as plt

    # Calculate average VADER positive and negative sentiment scores
    avg_vader_pos = sum(vader['pos'] for vader in df['vader_sentiment']) / len(df)
    avg_vader_neg = sum(vader['neg'] for vader in df['vader_sentiment']) / len(df)

    # Calculate average SentiWordNet positive and negative sentiment scores
    avg_sentiwordnet_pos = sum(sentiwordnet['pos'] for sentiwordnet in df['sentiwordnet_sentiment']) / len(df)
    avg_sentiwordnet_neg = sum(sentiwordnet['neg'] for sentiwordnet in df['sentiwordnet_sentiment']) / len(df)

    # Plotting
    fig, ax = plt.subplots(figsize=(8, 6))

    # VADER positive sentiment
    ax.bar(x=[0], height=avg_vader_pos, width=0.4, color='skyblue', label='VADER Pos')

    # VADER negative sentiment
    ax.bar(x=[0.5], height=avg_vader_neg, width=0.4, color='salmon', label='VADER Neg')

    # SentiWordNet positive sentiment
    ax.bar(x=[1.5], height=avg_sentiwordnet_pos, width=0.4, color='lightgreen', label='SentiWordNet Pos')

    # SentiWordNet negative sentiment
    ax.bar(x=[2], height=avg_sentiwordnet_neg, width=0.4, color='coral', label='SentiWordNet Neg')

    ax.set_title('Comparison of Average Positive and Negative Sentiment Scores')
    ax.set_xlabel('Sentiment Type')
    ax.set_ylabel('Average Sentiment Score')
    ax.set_xticks([0, 0.5, 1.5, 2])
    ax.set_xticklabels(['VP', 'VN', 'SP', 'SN'])
    ax.legend()
    plt.tight_layout()

    # Render the plot in Streamlit
    st.pyplot(fig)

    import nltk
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import sentiwordnet as swn
    from textblob import TextBlob
    import streamlit as st


    # Function to perform sentiment analysis with TextBlob
    def analyze_sentiment_textblob(sentence):
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        return {'polarity': polarity, 'subjectivity': subjectivity}


    # Function to perform sentiment analysis on each message in the DataFrame
    def analyze_sentiments(df):
        vader_sentiments = []
        sentiwordnet_sentiments = []
        textblob_sentiments = []
        for message in df['message']:
            vader_sentiment = analyze_sentiment_vader(message)
            sentiwordnet_sentiment = analyze_sentiment_sentiwordnet(message)
            textblob_sentiment = analyze_sentiment_textblob(message)
            vader_sentiments.append(vader_sentiment)
            sentiwordnet_sentiments.append(sentiwordnet_sentiment)
            textblob_sentiments.append(textblob_sentiment)
        return vader_sentiments, sentiwordnet_sentiments, textblob_sentiments


    # Example usage:
    # Assuming df is your DataFrame containing the messages
    vader_sentiments, sentiwordnet_sentiments, textblob_sentiments = analyze_sentiments(df)

    # Adding sentiment scores to the DataFrame
    df['vader_sentiment'] = vader_sentiments
    df['sentiwordnet_sentiment'] = sentiwordnet_sentiments
    df['textblob_sentiment'] = textblob_sentiments

    # Displaying the DataFrame with sentiment scores in Streamlit
    st.write("Sentiment Analysis Results:")
    st.write(df[['message', 'vader_sentiment', 'sentiwordnet_sentiment', 'textblob_sentiment']])

    import streamlit as st
    from nltk.sentiment import SentimentIntensityAnalyzer
    from nltk.corpus import sentiwordnet as swn
    from textblob import TextBlob
    import nltk


    # Function to perform sentiment analysis with VADER
    def analyze_sentiment_vader(sentence):
        sid = SentimentIntensityAnalyzer()
        sentiment = sid.polarity_scores(sentence)
        return sentiment


    # Function to perform sentiment analysis with SentiWordNet
    def analyze_sentiment_sentiwordnet(sentence):
        sentiment = {'pos': 0, 'neg': 0, 'obj': 0}
        tokens_count = 0
        for token in nltk.word_tokenize(sentence):
            synsets = list(swn.senti_synsets(token))
            if synsets:
                sentiment['pos'] += synsets[0].pos_score()
                sentiment['neg'] += synsets[0].neg_score()
                sentiment['obj'] += synsets[0].obj_score()
                tokens_count += 1
        if tokens_count > 0:
            sentiment['pos'] /= tokens_count
            sentiment['neg'] /= tokens_count
            sentiment['obj'] /= tokens_count
        return sentiment


    # Function to perform sentiment analysis with TextBlob
    def analyze_sentiment_textblob(sentence):
        blob = TextBlob(sentence)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        return {'polarity': polarity, 'subjectivity': subjectivity}


    # Perform sentiment analysis on messages in the DataFrame
    vader_sentiments = [analyze_sentiment_vader(message) for message in df['message']]
    sentiwordnet_sentiments = [analyze_sentiment_sentiwordnet(message) for message in df['message']]
    textblob_sentiments = [analyze_sentiment_textblob(message) for message in df['message']]

    import streamlit as st


    # Function to calculate the average sentiment scores for VADER
    def calculate_avg_vader_sentiment(vader_sentiments):
        return {
            'compound': sum(vader['compound'] for vader in vader_sentiments) / len(vader_sentiments),
            'neg': sum(vader['neg'] for vader in vader_sentiments) / len(vader_sentiments),
            'neu': sum(vader['neu'] for vader in vader_sentiments) / len(vader_sentiments),
            'pos': sum(vader['pos'] for vader in vader_sentiments) / len(vader_sentiments)
        }


    # Function to calculate the average sentiment scores for SentiWordNet
    def calculate_avg_sentiwordnet_sentiment(sentiwordnet_sentiments):
        return {
            'pos': sum(sentiwordnet['pos'] for sentiwordnet in sentiwordnet_sentiments) / len(sentiwordnet_sentiments),
            'neg': sum(sentiwordnet['neg'] for sentiwordnet in sentiwordnet_sentiments) / len(sentiwordnet_sentiments),
            'obj': sum(sentiwordnet['obj'] for sentiwordnet in sentiwordnet_sentiments) / len(sentiwordnet_sentiments)
        }


    # Function to calculate the average sentiment scores for TextBlob
    def calculate_avg_textblob_sentiment(textblob_sentiments):
        return {
            'polarity': sum(sentiment['polarity'] for sentiment in textblob_sentiments) / len(textblob_sentiments),
            'subjectivity': sum(sentiment['subjectivity'] for sentiment in textblob_sentiments) / len(
                textblob_sentiments)
        }


    # Calculate average sentiment scores
    avg_vader_sentiment = calculate_avg_vader_sentiment(vader_sentiments)
    avg_sentiwordnet_sentiment = calculate_avg_sentiwordnet_sentiment(sentiwordnet_sentiments)
    avg_textblob_sentiment = calculate_avg_textblob_sentiment(textblob_sentiments)

    # Display the average sentiment scores in a Streamlit app
    st.write("Average VADER Sentiment:")
    st.write(avg_vader_sentiment)

    st.write("Average SentiWordNet Sentiment:")
    st.write(avg_sentiwordnet_sentiment)

    st.write("Average TextBlob Sentiment:")
    st.write(avg_textblob_sentiment)

    # Selecting common sentiment attributes
    #average_textblob_sentiment.keys())

    # Average sentiment scores
    average_vader_sentiment = {'pos': 0.1256899182561309, 'neg': 0.03269073569482283, 'compound': 0.07002269754768403,
                               'neu': 0.7583311534968187}
    average_sentiwordnet_sentiment = {'pos': 0.19191135331516804, 'neg': 0.13022761126248866, 'obj': 3.5206403269754767}
    average_textblob_sentiment = {'polarity': 0.053023395898065095, 'subjectivity': 0.14325429385528646}

    # Extracting values for common attributes
  #
    #vader_values = [average_vader_sentiment[attr] for attr in common_attributes]
 #   sentiwordnet_values = [average_sentiwordnet_sentiment[attr] for attr in common_attributes]
  #  textblob_values = [average_textblob_sentiment[attr] for attr in common_attributes]

    # Plotting
   # fig, ax = plt.subplots()
    #x = range(len(common_attributes))

    # Plot bar charts for each sentiment analysis method
    #ax.bar(x, vader_values, width=0.25, label='VADER', align='center')
    #ax.bar(x, sentiwordnet_values, width=0.25, label='SentiWordNet', align='edge')
    #x.bar(x, textblob_values, width=0.25, label='TextBlob', align='edge')

    # Set labels and title
    #ax.set_xlabel('Sentiment Attribute')
    #ax.set_ylabel('Average Sentiment Score')
    #ax.set_title('Comparison of Average Sentiment Scores')
    #ax.set_xticks(x)
    #ax.set_xticklabels(common_attributes)
    #ax.legend()

    # Display the plot in Streamlit
    #st.pyplot(fig)

    import streamlit as st
    import matplotlib.pyplot as plt


    # Function to categorize polarity values into positive, negative, or neutral
    def categorize_polarity(polarity):
        if polarity > 0:
            return 'positive'
        elif polarity < 0:
            return 'negative'
        else:
            return 'neutral'


    # Initialize lists to store categorized polarity values for each method
    vader_polarities = {'positive': 0, 'negative': 0, 'neutral': 0}
    sentiwordnet_polarities = {'positive': 0, 'negative': 0, 'neutral': 0}
    textblob_polarities = {'positive': 0, 'negative': 0, 'neutral': 0}

    # Loop through each message and update polarity counts
    for message in df['message']:
        vader_sentiment = analyze_sentiment_vader(message)
        sentiwordnet_sentiment = analyze_sentiment_sentiwordnet(message)
        textblob_sentiment = analyze_sentiment_textblob(message)

        vader_polarity = vader_sentiment['compound']
        sentiwordnet_polarity = (sentiwordnet_sentiment['pos'] - sentiwordnet_sentiment['neg'])
        textblob_polarity = textblob_sentiment['polarity']

        vader_polarities[categorize_polarity(vader_polarity)] += 1
        sentiwordnet_polarities[categorize_polarity(sentiwordnet_polarity)] += 1
        textblob_polarities[categorize_polarity(textblob_polarity)] += 1

    # Plotting
    categories = ['positive', 'negative', 'neutral']
    vader_counts = [vader_polarities[cat] for cat in categories]
    sentiwordnet_counts = [sentiwordnet_polarities[cat] for cat in categories]
    textblob_counts = [textblob_polarities[cat] for cat in categories]

    fig, ax = plt.subplots()

    bar_width = 0.25
    index = range(len(categories))

    ax.bar(index, vader_counts, bar_width, label='VADER')
    ax.bar([i + bar_width for i in index], sentiwordnet_counts, bar_width, label='SentiWordNet')
    ax.bar([i + 2 * bar_width for i in index], textblob_counts, bar_width, label='TextBlob')

    ax.set_xlabel('Sentiment')
    ax.set_ylabel('Frequency')
    ax.set_title('Sentiment Analysis Comparison')
    ax.set_xticks([i + bar_width for i in index])
    ax.set_xticklabels(categories)
    ax.legend()

    # Display the plot in Streamlit
    st.pyplot(fig)

    import streamlit as st
    import pandas as pd
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Create a DataFrame containing the frequencies of sentiment classifications
    data = {
        'Sentiment': ['Positive', 'Negative', 'Neutral'],
        'VADER': vader_counts,
        'SentiWordNet': sentiwordnet_counts,
        'TextBlob': textblob_counts
    }
    df_heatmap = pd.DataFrame(data)

    # Set 'Sentiment' column as the index
    df_heatmap.set_index('Sentiment', inplace=True)

    # Plotting heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_heatmap, annot=True, cmap='coolwarm', fmt='g')
    plt.title('Sentiment Analysis Comparison')
    plt.xlabel('Sentiment Analysis Method')
    plt.ylabel('Sentiment')

    # Display the plot in Streamlit
    st.pyplot(plt)

    # Calculate polarity values
    vader_polarity = average_vader_sentiment.get('compound', 0.0)
    sentiwordnet_polarity = (average_sentiwordnet_sentiment.get('pos', 0.0) - average_sentiwordnet_sentiment.get('neg',
                                                                                                                 0.0)) / 2.0
    textblob_polarity = average_textblob_sentiment.get('polarity', 0.0)

    # Plotting
    methods = ['VADER', 'SentiWordNet', 'TextBlob']
    polarity_values = [vader_polarity, sentiwordnet_polarity, textblob_polarity]

    # Create a bar chart
    fig, ax = plt.subplots()
    ax.bar(methods, polarity_values, color=['blue', 'orange', 'green'])
    ax.set_xlabel('Sentiment Analysis Method')
    ax.set_ylabel('Average Polarity Value')
    ax.set_title('Comparison of Average Polarity Values')

    # Display the plot in Streamlit
    st.pyplot(fig)


    st.title("15. TRANSLATING FIRST 50 MESSAGES ACCORDING TO OUR CHOICE")

    import streamlit as st
    import matplotlib.pyplot as plt
    import pandas as pd
    import networkx as nx
    import community
    from googletrans import Translator


    # Function to translate messages from one language to another
    def translate_messages(messages, source_lang, target_lang):
        translator = Translator()
        translated_messages = []
        for message in messages:
            try:
                translation = translator.translate(message, src=source_lang, dest=target_lang)
                translated_messages.append(translation.text)
            except Exception as e:
                translated_messages.append(f"Translation error: {e}")
        return translated_messages


    # Set the source and target languages for translation
    source_lang = 'en'  # Assuming English is the source language
    target_lang = 'ja'  # Assuming Japanese is the target language

    # Select the first 30 messages
    first_50_messages = df['message'].head(50)

    # Translate the first 30 messages
    translated_messages = translate_messages(first_50_messages, source_lang, target_lang)

    # Create a DataFrame to store the original and translated messages
    translated_df = pd.DataFrame({'original_message': first_50_messages, 'translated_message': translated_messages})

    # Display the DataFrame with translated messages
    st.write(translated_df)

    # Define the source and target languages
    source_lang = 'en'  # Assuming English is the source language
    target_lang = 'hi'  # Assuming Hindi is the target language

    # Assuming you have a function translate_messages() that translates messages
    # Define the function if not already defined

    # Select the first 30 messages
    first_50_messages = df['message'].head(50)

    # Translate the first 30 messages
    translated_messages = translate_messages(first_50_messages, source_lang, target_lang)

    # Create a DataFrame to store the original and translated messages
    translated_df = pd.DataFrame({'Original Message': first_50_messages, 'Translated Message': translated_messages})

    # Display the DataFrame with translated messages in Streamlit
    st.write("Translated Messages:")
    st.dataframe(translated_df)
