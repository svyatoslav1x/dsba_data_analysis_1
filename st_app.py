import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import seaborn as sns
import streamlit as st
from scipy import stats

st.set_page_config(layout="wide")
sns.set_theme(style="whitegrid")


@st.cache_data
def load_and_clean_data():
    df = pd.read_csv("./data/sof_survey_2025.csv")

    critical_cols = [
        "ConvertedCompYearly",
        "WorkExp",
        "YearsCode",
        "EdLevel",
        "AISelect",
        "ICorPM",
    ]
    df_clean = df.dropna(subset=critical_cols).copy()

    cols_to_filter = ["ConvertedCompYearly", "ToolCountWork", "ToolCountPersonal"]
    for col in cols_to_filter:
        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)
        iqr = q3 - q1
        l = max(0, q1 - 1.5 * iqr)
        u = q3 + 1.5 * iqr

        df_clean = df_clean[
            ((df_clean[col] >= l) & (df_clean[col] <= u)) | df_clean[col].isna()
        ]

    df_clean["YearsCodeHobby"] = df_clean["YearsCode"] - df_clean["WorkExp"]
    df_clean["YearsCodeHobby"] = df_clean["YearsCodeHobby"].apply(
        lambda x: x if x >= 0 else 0
    )

    df_clean = df_clean[df_clean["YearsCodeHobby"] < 70]

    ed_mapping = {
        "Primary/elementary school": 1,
        "Secondary school (e.g. American high school, German Realschule or Gymnasium, etc.)": 2,
        "Some college/university study without earning a degree": 3,
        "Associate degree (A.A., A.S., etc.)": 4,
        "Bachelor’s degree (B.A., B.S., B.Eng., etc.)": 5,
        "Master’s degree (M.A., M.S., M.Eng., MBA, etc.)": 6,
        "Professional degree (JD, MD, Ph.D, Ed.D, etc.)": 7,
    }
    df_clean["EdLevel_Numeric"] = df_clean["EdLevel"].map(ed_mapping)
    df_clean.to_csv("cleaned_survey_data.csv", index=False)

    return df, df_clean


raw_df, full_df = load_and_clean_data()

st.sidebar.title("Filters")

available_roles = full_df["ICorPM"].dropna().unique().tolist()
selected_roles = st.sidebar.multiselect(
    "Roles", options=available_roles, default=available_roles
)

min_exp, max_exp = float(full_df["WorkExp"].min()), float(full_df["WorkExp"].max())
exp_range = st.sidebar.slider(
    "WorkExp", min_value=min_exp, max_value=max_exp, value=(min_exp, max_exp), step=1.0
)

filtered_df = full_df[
    (full_df["ICorPM"].isin(selected_roles))
    & (full_df["WorkExp"] >= exp_range[0])
    & (full_df["WorkExp"] <= exp_range[1])
].copy()

page_selection = st.sidebar.radio(
    "Navigation",
    [
        "Annotation and Description",
        "EDA",
        "Hypothesis 1",
        "Hypothesis 2",
        "Discussion",
        "API",
    ],
)

if page_selection == "Annotation and Description":
    st.markdown("""## Annotation

This project will try to explore the global landscape of the software development industry using data from the 2025 Stack Overflow Developer Survey. The primary purpose of the project is to analyze the relationships between a developer's experience, education level, adoption of AI tools, and their annual compensation. This will be achieved by performing exploratory data analysis and statistical analysis.

Project's team consisted of only one member: Eylkin Svyatoslav from group 252-1.


## Dataset Description

The subject area of the dataset is the technology and software development industry. The dataset is derived from the annual [Stack Overflow Developer Survey](https://github.com/StackExchange/Survey/tree/main), which captures responses from around 50k developers worldwide. The dataset contains numerous fields, including categorical fields (e.g., `EdLevel` for education, `AISelect` for AI usage stance, `ICorPM` for Individual Contributor vs. People Manager) and numerical fields (e.g., `ConvertedCompYearly` for USD salary, `WorkExp` for professional coding years, `YearsCode` for total coding years, `ToolsWork` for the number of distinct apps used at work, and `ToolsSide` for apps used on side projects). 
""")

elif page_selection == "EDA":
    st.markdown("""## EDA
To understand the distributions and basic relations between the key numerical fields: `ConvertedCompYearly`, `ToolCountWork`, `YearsCodeHobby`, `WorkExp`, were created some visualizations using four different plot types:

- **Histogram**: for `ConvertedCompYearly` to observe the distribution of global developer salaries (revealing a heavy right-skew).
- **Boxplot**: for `ToolCountWork` to visualize the spread, median, and variance of the number of distinct tools used at work, which also serves to visually verify our IQR outlier cleanup.
- **Scatter plot**: `YearsCodeHobby` vs `ConvertedCompYearly` to explore the variance and density of pre-professional coding experience and its initial lack of clear linear correlation with current salary.
- **Line plot**: `ConvertedCompYearly` over `WorkExp` to track the average salary trajectory and its confidence interval as professional experience increases over time.

Additionally key indicators were calculated and analysed.""")

    num_columns = [
        "ConvertedCompYearly",
        "WorkExp",
        "ToolCountWork",
        "ToolCountPersonal",
        "YearsCodeHobby",
        "YearsCode",
        "EdLevel_Numeric",
    ]
    stats_df = filtered_df[num_columns].describe().T[["mean", "50%", "std"]]
    stats_df.rename(columns={"50%": "median"}, inplace=True)
    st.dataframe(stats_df)

    fig1, ax1 = plt.subplots(figsize=(10, 6))
    sns.histplot(filtered_df["ConvertedCompYearly"], bins=40, ax=ax1)
    ax1.set_title("Salary Distribution (ConvertedCompYearly)", fontsize=14)
    ax1.set_xlabel("Yearly Compensation (USD)")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    fig2, ax2 = plt.subplots(figsize=(10, 4))
    sns.boxplot(x=filtered_df["ToolCountWork"], color="mediumseagreen", ax=ax2)
    ax2.set_title("ToolCountWork", fontsize=14)
    ax2.set_xlabel("Number of Tools")
    st.pyplot(fig2)

    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        x="YearsCodeHobby",
        y="ConvertedCompYearly",
        data=filtered_df,
        alpha=0.3,
        color="darkorange",
        s=30,
        ax=ax3,
    )
    ax3.set_title("Unprofessional Coding Years vs Salary", fontsize=14)
    ax3.set_xlabel("Years of Unprofessinal Coding")
    ax3.set_ylabel("Salary (USD)")
    st.pyplot(fig3)

    fig4, ax4 = plt.subplots(figsize=(10, 6))
    sns.lineplot(
        x="WorkExp",
        y="ConvertedCompYearly",
        data=filtered_df,
        color="purple",
        linewidth=2,
        ax=ax4,
    )
    ax4.set_title("Salary vs Work Experience", fontsize=14)
    ax4.set_xlabel("Professional Work Experience (Years)")
    ax4.set_ylabel("Salary (USD)")
    st.pyplot(fig4)

    fig5, ax5 = plt.subplots(figsize=(8, 6))
    corr_cols = [
        "ConvertedCompYearly",
        "WorkExp",
        "YearsCodeHobby",
        "ToolCountWork",
        "EdLevel_Numeric",
    ]
    corr_matrix = filtered_df[corr_cols].corr(method="spearman")
    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f",
        linewidths=0.5,
        ax=ax5,
    )
    st.pyplot(fig5)

    fig6, ax6 = plt.subplots(figsize=(10, 6))
    role_df = filtered_df.dropna(subset=["ICorPM", "EdLevel_Numeric"])
    sns.barplot(
        x="EdLevel_Numeric",
        y="ConvertedCompYearly",
        hue="ICorPM",
        data=role_df,
        estimator=np.median,
        palette="viridis",
        errorbar=None,
        ax=ax6,
    )
    ax6.set_title("Median Salary by Education Level and Role", fontsize=14)
    ax6.set_xlabel("Education Level (1=Primary, 7=Professional)")
    ax6.set_ylabel("Median Salary (USD)")
    ax6.legend(title="Role")
    st.pyplot(fig6)

    fig7, ax7 = plt.subplots(figsize=(12, 7))
    sns.scatterplot(
        x="WorkExp",
        y="ConvertedCompYearly",
        hue="AISelect",
        data=filtered_df,
        alpha=0.6,
        palette="Set1",
        s=40,
        ax=ax7,
    )
    ax7.set_xlabel("Professional Work Experience (Years)")
    ax7.set_ylabel("Yearly Compensation (USD)")
    ax7.legend(title="AI Usage Stance", bbox_to_anchor=(1.05, 1), loc="upper left")
    st.pyplot(fig7)

elif page_selection == "Hypothesis 1":
    st.markdown("""## Hypothesis 1

A common assumption is that AI makes developers more valuable.
* **Observation:** Observation 1 shows that median salaries visually appear higher for non-AI users across both Individual Contributors and People Managers.
* **Null Hypothesis (H0):** There is no significant difference in the median annual compensation between daily AI users and non-users within a given role.
* **Test Chosen:** Mann-Whitney U test. As shown by the initial Salary Histogram, compensation is heavily right-skewed (non-normal). Therefore, we use a non-parametric test to compare the distributions of two independent groups.""")

    ai_users = filtered_df[filtered_df["AISelect"] == "Yes, I use AI tools daily"]
    non_ai_users = filtered_df[filtered_df["AISelect"] == "No, and I don't plan to"]

    ic_ai = ai_users[ai_users["ICorPM"] == "Individual contributor"][
        "ConvertedCompYearly"
    ].dropna()
    ic_non = non_ai_users[non_ai_users["ICorPM"] == "Individual contributor"][
        "ConvertedCompYearly"
    ].dropna()

    st.write("Individual Contributors")
    if len(ic_ai) > 0 and len(ic_non) > 0:
        st.write(
            f"Median Salary | Non-User: ${ic_non.median():,.0f} vs AI Daily: ${ic_ai.median():,.0f}"
        )
        stat_ic, p_ic = stats.mannwhitneyu(ic_non, ic_ai, alternative="greater")
        st.write(f"p-value: {p_ic:.4e}")
        if p_ic < 0.05:
            st.write(
                "Conclusion: Reject H0. Non-using ICs earn a statistically significant premium over AI-using ICs."
            )
        else:
            st.write("Conclusion: Fail to reject H0.")

    pm_ai = ai_users[ai_users["ICorPM"] == "People manager"][
        "ConvertedCompYearly"
    ].dropna()
    pm_non = non_ai_users[non_ai_users["ICorPM"] == "People manager"][
        "ConvertedCompYearly"
    ].dropna()

    st.write("People Managers")
    if len(pm_ai) > 0 and len(pm_non) > 0:
        st.write(
            f"Median Salary | Non-User: ${pm_non.median():,.0f} vs AI Daily: ${pm_ai.median():,.0f}"
        )
        stat_pm, p_pm = stats.mannwhitneyu(pm_non, pm_ai, alternative="greater")
        st.write(f"p-value: {p_pm:.4e}")
        if p_pm < 0.05:
            st.write(
                "Conclusion: Reject H0. Non-using Managers earn a statistically significant premium over AI-using Managers."
            )
        else:
            st.write("Conclusion: Fail to reject H0.")

    fig_h1, ax_h1 = plt.subplots(figsize=(10, 6))
    plot_h1 = filtered_df[
        (
            filtered_df["AISelect"].isin(
                ["Yes, I use AI tools daily", "No, and I don't plan to"]
            )
        )
        & (filtered_df["ICorPM"].isin(["Individual contributor", "People manager"]))
    ].copy()
    plot_h1["AI Usage"] = plot_h1["AISelect"].map(
        {"Yes, I use AI tools daily": "Daily AI", "No, and I don't plan to": "Non-User"}
    )

    sns.boxplot(
        data=plot_h1,
        x="ICorPM",
        y="ConvertedCompYearly",
        hue="AI Usage",
        palette="pastel",
        showfliers=False,
        ax=ax_h1,
    )
    sns.stripplot(
        data=plot_h1,
        x="ICorPM",
        y="ConvertedCompYearly",
        hue="AI Usage",
        dodge=True,
        palette="dark:black",
        alpha=0.25,
        jitter=True,
        size=3,
        ax=ax_h1,
    )

    handles, labels = ax_h1.get_legend_handles_labels()
    ax_h1.legend(handles[:2], labels[:2], title="AI Usage")
    ax_h1.set_title(
        "Observation 1: The Non-Adopter Salary Difference by Role", fontsize=14
    )
    ax_h1.set_ylabel("Salary (USD)")
    ax_h1.set_xlabel("Role")
    st.pyplot(fig_h1)

    st.markdown(
        """The data proved the opposite of the initial hypothesis. **Refusing to use AI tools provides a statistically significant salary premium** for both Individual Contributors and People Managers. """
    )

elif page_selection == "Hypothesis 2":
    st.markdown("""## Hypothesis 2

* **Observation:** Observation 2 shows a visual "flip" (interaction effect): daily AI usage correlates with lower pay for standard hobbyists (<= 8 yrs), but correlates with higher pay for hardcore hobbyists (> 8 yrs).
* **Null Hypothesis (H0):** Pre-professional hobby experience does not alter the salary impact of AI usage (i.e., AI users do not earn significantly more or less than non-users in either hobbyist tier).
* **Test Chosen:** Mann-Whitney U test.""")

    ai_df = filtered_df[
        filtered_df["AISelect"].isin(
            ["Yes, I use AI tools daily", "No, and I don't plan to"]
        )
    ].copy()

    std_hobby = ai_df[ai_df["YearsCodeHobby"] <= 8]
    ai_std = std_hobby[std_hobby["AISelect"] == "Yes, I use AI tools daily"][
        "ConvertedCompYearly"
    ].dropna()
    non_std = std_hobby[std_hobby["AISelect"] == "No, and I don't plan to"][
        "ConvertedCompYearly"
    ].dropna()

    st.write("Standard Hobbyists (<= 8 Years Hobby Coding)")
    if len(ai_std) > 0 and len(non_std) > 0:
        st.write(
            f"Median Salary | Non-User: ${non_std.median():,.0f} vs AI Daily: ${ai_std.median():,.0f}"
        )
        stat_std, p_std = stats.mannwhitneyu(non_std, ai_std, alternative="greater")
        st.write(f"p-value: {p_std:.4e}")
        if p_std < 0.05:
            st.write("Conclusion: Reject H0. Non-users earn more.")
        else:
            st.write("Conclusion: Fail to reject H0.")

    hdc_hobby = ai_df[ai_df["YearsCodeHobby"] > 8]
    ai_hard = hdc_hobby[hdc_hobby["AISelect"] == "Yes, I use AI tools daily"][
        "ConvertedCompYearly"
    ].dropna()
    non_hard = hdc_hobby[hdc_hobby["AISelect"] == "No, and I don't plan to"][
        "ConvertedCompYearly"
    ].dropna()

    st.write("Hardcore Hobbyists (> 8 Years Hobby Coding)")
    if len(ai_hard) > 0 and len(non_hard) > 0:
        st.write(
            f"Median Salary | Non-User: ${non_hard.median():,.0f} vs AI Daily: ${ai_hard.median():,.0f}"
        )
        stat_hard, p_hard = stats.mannwhitneyu(ai_hard, non_hard, alternative="greater")
        st.write(f"p-value: {p_hard:.4e}")
        if p_hard < 0.05:
            st.write("Conclusion: Reject H0. AI users earn more.")
        else:
            st.write("Conclusion: Fail to reject H0.")

    fig_h2, ax_h2 = plt.subplots(figsize=(9, 6))
    plot_h2 = ai_df.copy()
    plot_h2["Hobby_Background"] = plot_h2["YearsCodeHobby"].apply(
        lambda x: "Hardcore (>8 yrs)" if x > 8 else "Standard (<=8 yrs)"
    )
    plot_h2["AI_Usage"] = plot_h2["AISelect"].map(
        {
            "Yes, I use AI tools daily": "Daily AI User",
            "No, and I don't plan to": "Non-User",
        }
    )

    sns.pointplot(
        data=plot_h2,
        x="Hobby_Background",
        y="ConvertedCompYearly",
        hue="AI_Usage",
        estimator=np.median,
        errorbar=("ci", 95),
        capsize=0.1,
        palette="Set1",
        dodge=True,
        markers=["o", "s"],
        ax=ax_h2,
    )
    ax_h2.set_title(
        "H2: The 'Crutch vs Accelerator' Interaction Effect",
        fontsize=14,
        fontweight="bold",
    )
    ax_h2.set_xlabel("Hobby Coding Background")
    ax_h2.set_ylabel("Median Salary (USD)")
    ax_h2.legend(title="AI Usage")
    ax_h2.set_title(
        "Observation 2: The 'Crutch vs Accelerator' Interaction Effect", fontsize=14
    )
    ax_h2.grid(axis="y", linestyle="--", alpha=0.7)
    st.pyplot(fig_h2)

    st.markdown("""The data showed that:  
- For developers with a standard background (<= 8 years of pre-professional hobby coding), daily AI usage is associated with significantly *lower* pay, suggesting it may act as a "crutch" for less experienced developers.
- For hardcore hobbyists, the difference between earnings of daily ai users and non-users is not statistically significant. Although, it shows a strong trend that AI acts as an accelerator for senior devs, reversing the 'crutch' penalty seen in junior hobbyists.""")

elif page_selection == "Discussion":
    st.markdown("""
## Discussion

**What I did and why:**
- Performed data cleanup by dropping missing values in critical columns, and applying the IQR method to remove extreme outliers.
- Computed descriptive stats (mean, median, std) for 4+ numerical fields and created multiple plot types (histograms, boxplots, line plots, scatterplots) for initial exploration of distributions.
- Added 2 transformed columns: `YearsCodeHobby` (subtracting professional experience from total coding years) and `EdLevel_Numeric` (ordinal mapping of education levels) to enable categorical grouping and quantitative hypothesis testing.
- For detailed overview: used comparisons with hue, side-by-side subplot visualizations, and a Spearman correlation heatmap to reveal interactions (e.g., work vs. personal tool usage, correlation between experience and pay).
- Tested 2 hypotheses
- All plots were created using Matplotlib/Seaborn.
    """)
    csv = filtered_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download cleaned_data.csv",
        data=csv,
        file_name="cleaned_survey_data.csv",
        mime="text/csv",
    )

elif page_selection == "API":
    st.title("API Data Submission")
    with st.form("api_post_form", clear_on_submit=True):
        new_salary = st.number_input(
            "Annual Salary (USD)", min_value=0, value=85000, step=5000
        )
        new_exp = st.number_input(
            "Years Pro Experience", min_value=0.0, value=3.0, step=1.0
        )
        new_hobby = st.number_input(
            "Years Hobby Coding", min_value=0.0, value=2.0, step=1.0
        )
        new_role = st.selectbox(
            "Current Role", ["Individual contributor", "People manager"]
        )
        new_ai = st.selectbox(
            "AI Usage", ["Yes, I use AI tools daily", "No, and I don't plan to"]
        )
        new_tools = st.number_input(
            "Distinct Tools Used at Work", min_value=0.0, value=6.0, step=1.0
        )

        submitted = st.form_submit_button(
            "Send POST Request to API", use_container_width=True
        )

        if submitted:
            payload = {
                "ConvertedCompYearly": new_salary,
                "WorkExp": new_exp,
                "YearsCodeHobby": new_hobby,
                "EdLevel_Numeric": 5.0,
                "ToolCountWork": new_tools,
                "ToolCountPersonal": 3.0,
                "AISelect": new_ai,
                "ICorPM": new_role,
            }
            try:
                res = requests.post(
                    "https://svelkin-dsba-data-analysis-project-ziq7.onrender.com/api/developers",
                    json=payload,
                )
                if res.status_code == 201:
                    st.success("Successfully added via API.")
                else:
                    st.error("API Error")
            except requests.exceptions.ConnectionError:
                st.error("Connection Failed. Ensure API is running.")
