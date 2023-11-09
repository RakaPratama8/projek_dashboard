# %%
import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
# from babel.numbers import format_currency
sns.set(style="dark")

# %%
df = pd.read_csv("all_df.csv")

# %%
df.head(10)

# %%
df.info()

# %%
df['datetime'] = pd.to_datetime(df['datetime'])

# %% [markdown]
# # EDA

# %%
df.groupby(by=['station']).agg({
    'PM2.5' : ['mean', 'max', 'min'],
})

# %%
df.groupby(by=['station', 'year', 'month']).agg({
    'PM2.5' : ['mean', 'max', 'min'],
    'PM10' : ['mean', 'max', 'min']
})

# %%
df.groupby(by=['day']).agg({
    'PM2.5' : 'mean'
})

# %%
df.groupby(by=['station', 'datetime']).agg({
    'TEMP' : 'mean',
    'PM2.5' : 'mean',
    'PM10' : 'mean',
})

# %%
df.groupby(by='wd').agg({
    'PM2.5' : ['mean', 'max', 'min'],
    'PM10' : ['mean', 'max', 'min']
})

# %%
df.groupby(by=['station', 'wd']).agg({
    'wd' : 'count'
}).head(15)

# %%
df.groupby(by=['station', 'datetime']).agg({
    'TEMP' : 'mean',
    'PM2.5' : 'mean',
    'PM10' : 'mean',
}).reset_index()

# %% [markdown]
# # DASHBOARD

# %%
def create_daily_average_PM(df):
    daily_average_PM = df.resample(rule="D", on='datetime').agg({
        'PM2.5' : 'mean'
    }).reset_index()
    
    return daily_average_PM

def create_by_state_PM10(df):
    daily_PM10_by_state = df.groupby(by="station").agg({
        'PM10' : 'mean'
    }).reset_index()
    
    return daily_PM10_by_state

def create_by_state_PM25(df):
    daily_PM25_by_state = df.groupby(by="station").agg({
        'PM2.5' : 'mean'
    }).reset_index()
    
    return daily_PM25_by_state

def create_average_temp_PM(df):
    average_temp_PM = df.groupby(by='TEMP').agg({
        'PM2.5' : 'mean',
        'PM10' : 'mean',
    }).reset_index()
    
    return average_temp_PM

# %%
min_date = df["datetime"].min()
max_date = df["datetime"].max()

with st.sidebar:
    st.image("img.jpeg")

    start_date, end_date = st.date_input(
        label="Time Range", min_value=min_date,
        max_value=max_date,value=[min_date, max_date]
    )

# %%
main_df = df[(df["datetime"] >= str(start_date))&
                (df["datetime"] <= str(end_date))]

daily_average_pm = create_daily_average_PM(main_df)
daily_average_PM10_by_state = create_by_state_PM10(main_df)
daily_average_PM25_by_state = create_by_state_PM25(main_df)
daily_average_temp_pm = create_average_temp_PM(main_df)

# %%
st.header("AIR QUALITY INDEX :sparkles")

# %%
st.subheader("Daily PM2.5 Intensity")
fig, ax = plt.subplots(figsize=(16,8))

ax.plot(
    daily_average_pm['datetime'],
    daily_average_pm['PM2.5'],
    marker='o',
    linewidth=2,
    markerfacecolor='blue',
)

ax.set_ylabel('PM2.5', fontsize=30)
ax.set_xlabel("Datetime", fontsize=30)
ax.tick_params(axis='x', labelsize=15)
ax.tick_params(axis='y', labelsize=20)
ax.grid()


st.pyplot(fig)

# %%
st.subheader("Top 5 state with highest pollution")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))

colors = ["#6F61C0", "#7091F5", "#7091F5", "#7091F5", "#7091F5"]

sns.barplot(data=daily_average_PM10_by_state.head(5).sort_values(by='PM10', ascending=False), x='station', y='PM10', palette=colors, ax=ax[0])
ax[0].set_ylabel('PM10', fontsize=30)
ax[0].set_xlabel("Station", fontsize=30)
ax[0].set_title("Top 5 Highest PM10 by state", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)

sns.barplot(data=daily_average_PM25_by_state.head(5).sort_values(by='PM2.5', ascending=False), x='station', y='PM2.5', palette=colors, ax=ax[1])
ax[1].set_ylabel('PM2.5', fontsize=30)
ax[1].set_xlabel("Station", fontsize=30)
ax[1].set_title("Top 5 Highest PM2.5 by state", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)

st.pyplot(fig)

# %%
st.subheader("Top 5 state with lowest pollution")
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(35, 15))

colors = ["#6F61C0", "#7091F5", "#7091F5", "#7091F5", "#7091F5"]

sns.barplot(data=daily_average_PM10_by_state.tail(5).sort_values(by='PM10', ascending=True), x='station', y='PM10', palette=colors, ax=ax[0])
ax[0].set_ylabel('PM10', fontsize=30)
ax[0].set_xlabel("Station", fontsize=30)
ax[0].set_title("Top 5 Lowest PM10 by state", loc="center", fontsize=50)
ax[0].tick_params(axis='y', labelsize=35)
ax[0].tick_params(axis='x', labelsize=30)

sns.barplot(data=daily_average_PM25_by_state.tail(5).sort_values(by='PM2.5', ascending=True), x='station', y='PM2.5', palette=colors, ax=ax[1])
ax[1].set_ylabel('PM2.5', fontsize=30)
ax[1].set_xlabel("Station", fontsize=30)
ax[1].set_title("Top 5 Lowest PM2.5 by state", loc="center", fontsize=50)
ax[1].tick_params(axis='y', labelsize=35)
ax[1].tick_params(axis='x', labelsize=30)

st.pyplot(fig)

# %%
st.subheader('Do changes in temperature affect the intensity of air pollution?')
fig, ax = plt.subplots(figsize=(35, 15))

ax.plot(
    daily_average_temp_pm['TEMP'],
    daily_average_temp_pm['PM2.5'],
    linewidth=5,
    marker='.',
    markerfacecolor='blue',
    label='PM2.5',
)
ax.plot(
    daily_average_temp_pm['TEMP'],
    daily_average_temp_pm['PM10'],
    linewidth=5,
    marker='.',
    markerfacecolor='red',
    label='PM10',
)

plt.legend(fontsize=30)
ax.tick_params(axis='y', labelsize=35)
ax.tick_params(axis='x', labelsize=30)
ax.set_xlabel('Temperature', fontsize=25)

st.pyplot(fig)

# %%
col1, col2, col3, col4 = st.columns(4)

with col1:
    SO2_meter = df['SO2'].mean()
    st.metric("Average SO2", value=SO2_meter)

with col2:
    NO2_meter = df['NO2'].mean()
    st.metric("Average NO2", value=NO2_meter)

with col3:
    CO_meter = df['CO'].mean()
    st.metric("Average CO", value=CO_meter)

with col4:
    O3_meter = df['O3'].mean()
    st.metric("Average O3", value=O3_meter)


