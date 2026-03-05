# HR Check-ins Analytics Tool — User Guide

---

## Table of Contents

1. [Overview](#1-overview)
2. [Getting Started](#2-getting-started)
3. [Home Page](#3-home-page)
4. [File Requirements](#4-file-requirements)
5. [Using the KPIs Section](#5-using-the-kpis-section)
6. [Employee KPIs — Detailed Reference](#6-employee-kpis--detailed-reference)
7. [Manager KPIs — Detailed Reference](#7-manager-kpis--detailed-reference)
8. [Employee & Manager Combined KPIs — Detailed Reference](#8-employee--manager-combined-kpis--detailed-reference)
9. [Using the Compare (Year-over-Year) Section](#9-using-the-compare-year-over-year-section)
10. [NLP — Open-Ended Answer Analysis](#10-nlp--open-ended-answer-analysis)
11. [Downloads & Exports](#11-downloads--exports)
12. [How Name Matching Works](#12-how-name-matching-works)
13. [Error Messages — What They Mean and How to Fix Them](#13-error-messages--what-they-mean-and-how-to-fix-them)
14. [Frequently Asked Questions](#14-frequently-asked-questions)
15. [Glossary](#15-glossary)

---

## 1. Overview

### What is this tool?

The **HR Check-ins Analytics Tool** is a web-based application built for the HR team. It takes the raw Employee and Manager Check-In survey spreadsheets, cleans and standardises the data against the **Mena Report** (the official employee directory), and then produces a comprehensive set of KPIs, charts, and downloadable reports — all inside your browser.

### What does it do?

| Capability | Description |
|-----------|-------------|
| **Data cleaning** | Matches survey respondents to their official (canonical) names from the Mena Report using email addresses |
| **Employee KPIs** | Goal alignment, stress analysis, pulse-check requests, manager behaviors, work culture, department dynamics, company resources, and more |
| **Manager KPIs** | At-risk employees, promotion readiness (with review dates), culture fit, department dynamics, stress assessment, and more |
| **Combined KPIs** | Side-by-side employee vs. manager views for job changes, adaptability, check-in meeting frequency, reward & recognition, mistakes handling, employee input, and team integration |
| **Year-over-Year Comparison** | Upload two years of data and see delta indicators for every KPI — what improved and what declined |
| **NLP Analysis** | AI-powered analysis of open-ended survey answers to detect themes, sentiment, severity, and actionable recommendations (requires a GitHub Models token) |
| **Department filtering** | All KPIs can be filtered to a specific department at any time |
| **Downloads** | CSV and PNG exports are available for nearly every table and chart |

### Who is this for?

This guide is for **HR team members** who will use the tool day-to-day. No technical knowledge is required — you only need your survey files and the Mena Report.

---

## 2. Getting Started

### Accessing the tool

1. Open your web browser (Chrome, Edge, or Firefox recommended).
2. Navigate to the URL provided by your IT team (e.g. `http://localhost:8501` for a local installation).
3. The **Home** page will load.

### Quick-start checklist

Before you begin, make sure you have the following files ready:

- [ ] **Mena Report** for the year you are analysing (e.g. `Mena Report 2025.xlsx`)
- [ ] **Employee Check-In** survey export (e.g. `Employee Check-In 2025.xlsx`)
- [ ] **Manager Check-In** survey export (e.g. `Manager Check-In 2025.xlsx`)

If you plan to use the **Compare** section, you will need a second set of all three files for the other year.

---

## 3. Home Page

When the tool loads, you will see the **Talent Management** home page with three cards:

| Card | Status | Description |
|------|--------|-------------|
| **Check-ins** | Active | Click **Open Check-ins** to access cleaning, KPIs, comparison, and NLP analysis |
| **Performance Appraisals (PA)** | Closed | Reserved for future PA score analysis (not yet available) |
| **Employee Turnover Prediction** | Closed | Reserved for future predictive retention analytics (not yet available) |

Click **Open Check-ins** to proceed.

### Sidebar navigation

Once inside Check-ins, the left sidebar lets you switch between:

- **KPIs** — Analyse a single year of check-in data
- **Compare** — Compare two years side-by-side

When **KPIs** is selected, an additional **NLP** checkbox appears under "Subsections". Enable it to access the open-ended answer analysis module.

A **Back to Home** button is always visible at the top of the page.

---

## 4. File Requirements

### 4.1 Accepted file formats

| Format | Extension | Notes |
|--------|-----------|-------|
| Excel (recommended) | `.xlsx` | Best compatibility |
| Legacy Excel | `.xls` | Supported but `.xlsx` preferred |
| CSV | `.csv` | Comma-separated values |

### 4.2 File content requirements

#### Employee Check-In must contain:

| Required Column | Purpose |
|----------------|---------|
| An **email** column (e.g. "Your Work Email Address", "Email", or any column containing the word "email") | Used to match employees to the Mena Report |
| **Your Manager's Name** (exact header) | Identifies the reporting manager for each employee |
| A **Timestamp** column (e.g. "Timestamp", "Submission Timestamp", "Date") | Used to determine the survey year automatically |

The tool also automatically detects many optional question columns by keyword matching (e.g. columns containing "stress", "goals", "promotion", etc.). You do not need to rename these — the tool finds them.

#### Manager Check-In must contain:

| Required Column | Purpose |
|----------------|---------|
| An **email** column | Used to match managers to the Mena Report |
| **Subordinate Name** (exact header) | Identifies the employee being evaluated |
| A **Timestamp** column | Used to determine the survey year automatically |

#### Mena Report must contain:

| Required Column | Purpose |
|----------------|---------|
| **Employee Name** | The canonical (official) employee name — the single source of truth |
| **Email** | Used to match survey responses to employees |
| **Manager Name** | Used to canonicalise subordinate names in the Manager Check-In |

### 4.3 Year detection

The tool extracts the year from the **Timestamp** column inside each file (not from the filename). It checks that the Employee Check-In and Manager Check-In come from the same year. In the Compare section, it also verifies that the Mena Report filename includes a year matching the check-in data year.

### 4.4 Auto-detection of Mena Report

If you place a Mena Report file in the `Data/` folder of the tool's directory, the tool will auto-detect it. This means you can skip uploading the Mena Report manually in the KPIs section. The auto-detection looks for files whose name contains both "mena" and "report" (case-insensitive). If multiple files match, the most recently modified one is used.

> **Tip:** For the Compare section, you must always upload the Mena Report for each period explicitly.

---

## 5. Using the KPIs Section

This section analyses **one year** of check-in data.

### Step-by-step

1. Select **KPIs** from the sidebar.
2. Open the **Upload** expander at the top.
3. Upload your files:
   - **Mena Report** — upload it or let the tool auto-detect it from the `Data/` folder.
   - **Employee Check-In** — the employee survey export.
   - **Manager Check-In** — the manager survey export.
4. (Optional) Select a department from the **Filter by department** dropdown. The dropdown is populated from the "Company Name / Department" column in your data. Choose "All departments" to see the full organisation.
5. Click the **Run** button.
6. If any issues are found, a single red error box lists **all problems at once**, grouped by category. Fix them and click Run again.
7. Once successful, a row count appears (e.g. "Employee rows: 45 | Manager rows: 52").
8. Use the **View** radio buttons to switch between:
   - **Employee** — KPIs derived from employee survey responses
   - **Manager** — KPIs derived from manager survey responses
   - **Employee & Manager** — Combined views comparing both perspectives

### What happens behind the scenes

1. Employee emails are matched to the Mena Report to find each respondent's canonical name. A new column `Mena Name` is added.
2. Manager emails are matched similarly. A new column `Name on Mena` is added.
3. Subordinate names in the Manager Check-In are canonicalised by looking up the employee under that manager in the Mena Report.
4. A `Match Source` column is added to track how each match was made (see [Section 12](#12-how-name-matching-works)).
5. All KPI calculations use the cleaned, name-matched data.

---

## 6. Employee KPIs — Detailed Reference

Each KPI appears inside a collapsible **expander**. Click the title to open or close it.

### 6.1 Alignment on Goals with Managers

**What it shows:** Two side-by-side panels:

| Left panel | Right panel |
|-----------|------------|
| How many employees feel **aligned** with their manager on the department's goals vs. how many do **not** | How many employees **discussed** their professional goals with their manager this year vs. how many did **not** |

Both panels show the total number of respondents and break the count into Yes/No.

### 6.2 Alignment of Responsibility with Professional Growth

**What it shows:** Whether employees feel their day-to-day tasks and responsibilities support their desired career growth.

- Left side: counts of **Aligned** vs. **Not Aligned**.
- Right side: for every employee who answered "No", the tool lists their name and their written elaboration (if provided).

### 6.3 Managers Behaviors (Venn Diagram)

**What it shows:** A Venn diagram classifying employees into three groups based on how they described their manager's behaviors:

| Group | Meaning |
|-------|---------|
| **Left circle (gray)** | Employees who selected only **positive** statements (e.g. "provides clear guidance", "gives recognition") |
| **Right circle (red)** | Employees who selected only **negative** statements (e.g. "provides insufficient information", "rarely gives credit") |
| **Overlap** | Employees who selected **both** positive and negative statements |

Below the diagram, a "Cases to Review" section lists:
- Employees who selected **only negative** statements
- Employees who selected **both** positive and negative statements

These are names the HR team may want to follow up on.

### 6.4 Work Culture and Environment — Supportive Work Environment

**What it shows:** Two panels side-by-side:

- **Left:** Yes/No count — how many employees feel the company's culture fosters a collaborative and supportive work environment. Names of employees who answered "No" are listed.
- **Right:** The top 3 most-selected options from the "describe the company's work culture" multi-select question.

### 6.5 Enhancing Department Dynamics — Employees

**What it shows:** A horizontal bar chart counting how many employees selected each option for how department dynamics could be enhanced. Options typically include things like improved communication, better tools, team-building, etc.

### 6.6 Employee Stress

**What it shows:** Two tabs:

| Tab | Content |
|-----|---------|
| **Frequency** | Three cards showing how many employees chose each stress level: *Extremely frequent*, *Frequent*, or *Less frequent* |
| **Reasons (Who mentioned what)** | A table listing each stress reason, along with the names of employees who selected it and their frequency level. Only employees who chose *Extremely frequent* or *Frequent* are included. |

### 6.7 Pulse-Check Meeting

**What it shows:** Two tabs:

| Tab | Content |
|-----|---------|
| **Overview** | How many employees said **Yes** vs. **No** to needing more pulse-check meetings with HR |
| **Reasons (YES only)** | For employees who said Yes, what reasons they selected (with counts) |

### 6.8 Company Resources

**What it shows:** A table listing all company resources or practices that employees use to ease their work experience, with a count for each option. If any employee selected "Other", the freeform text entries are shown below the table.

### 6.9 Additional Employee KPIs (Keyword-Based)

**What it shows:** A set of compact metric cards showing percentages:

| Card | What it measures |
|------|------------------|
| Job requirements changed | % of employees who encountered changes in their job requirements |
| Adapted well (4–5) | % of employees who rated their adaptability 4 or 5 on a 5-point scale |
| Tasks aligned with growth | % of employees who feel their tasks are aligned with professional growth |
| Manager considers input | % of employees who feel their manager seeks and considers their input |
| HR pulse request rate | % of employees requesting more HR pulse-check meetings |
| Recommend company | % of employees who would recommend working at the company |
| Stress rate (Employees) | % of employees reporting *Frequent* or *Extremely frequent* stress |

---

## 7. Manager KPIs — Detailed Reference

### 7.1 Additional Manager KPIs (Yes %)

**What it shows:** Compact metric cards — for each card, the count and percentage of subordinates for whom the manager answered "Yes":

| Card | What it measures |
|------|------------------|
| Encountered job changes | Manager confirmed the employee encountered changes in job requirements |
| At risk of low performance | Manager flagged the employee as at risk of low performance |
| Better in another dept | Manager believes the employee would perform better in a different department |
| Ready for promotion | Manager considers the employee ready for promotion today |
| Seeks team input | Manager confirms they actively seek and consider the employee's input |
| Fits in company culture | Manager confirms the employee fits in the company culture |
| Manager Stress rate | % of responses where the manager rated the employee's (or their own) stress as *Frequent* or *Extremely frequent* |

### 7.2 At Risk of Low Performance

**What it shows:** A detailed table of employees flagged at risk, with columns:

| Column | Description |
|--------|-------------|
| Employee Name | The subordinate's canonical name |
| At Risk of Low Performance | Always "Yes" for rows shown |
| Reason | The manager's written reason for the flag |
| PIP | Whether a Performance Improvement Plan was recommended (Yes/No) |
| Reason if no | If PIP is "No", the manager's explanation of why not |

A CSV download button is provided.

### 7.3 Manager Insights — Promotion

Two nested sections:

#### Ready for promotion (Yes)

A table of employees marked as promotion-ready, with the position stated by the manager (if provided). Includes a name filter and a CSV download button.

#### Not ready for promotion (No) — Suggested review dates

Employees marked as not ready are grouped by the **review date** the manager suggested, organised by month. Each month appears as a sub-expander with the list of names. Additional tools:

- **Diagnostics:** Shows entries with missing or unparsable review dates so you can follow up.
- **Find a specific name:** Type a name to see its raw and parsed review date.

### 7.4 Employee — Company Culture Fit (Manager)

**What it shows:** Two side-by-side metrics: how many employees fit vs. don't fit in the company's culture, according to their manager.

### 7.5 Employee Contribution to Department Dynamics — Managers

**What it shows:** A horizontal bar chart of how managers rated each employee's contribution to enhancing department dynamics. Similar layout to the employee dynamics chart, but from the manager's perspective.

### 7.6 Stress Frequency — Managers

**What it shows:** Three columns showing how managers assessed employee stress:

| Column | Meaning |
|--------|---------|
| *EXTREMELY FREQUENT* | Count + top 3 stress reasons |
| *FREQUENT* | Count + top 3 stress reasons |
| *LESS FREQUENT* | Count + top 3 stress reasons |

---

## 8. Employee & Manager Combined KPIs — Detailed Reference

These KPIs show **both perspectives side-by-side** so you can see where employees and managers agree or diverge.

### 8.1 Changes in Job Responsibilities (Employee vs Manager)

- Top: how many employees said "Yes" vs. how many managers said "Yes" to job responsibility changes.
- Bottom: a grouped bar chart comparing the **types of changes** selected by employees vs. managers (e.g. new responsibilities added, tasks removed, role restructured).

### 8.2 Adapting to Change (Employee vs Manager)

- Shows how many employees rated themselves 4–5 (able to adapt) vs. how many managers rated those employees 4–5.
- Also shows how many rated 1–2 (unable to adapt).
- If no one scored 1–2, a green bullet confirms this.

### 8.3 Check-in Meeting Frequency

- A vertical grouped bar chart comparing the percentage of employees vs. managers who reported each meeting frequency category (e.g. "Weekly", "Biweekly", "Monthly", "Few", "Zero").
- Below the chart: a list of employees where **both** the employee and their manager reported having **fewer than one meeting per month**. These are names to follow up on.

### 8.4 Reward & Recognition (Employee vs Manager)

- A table listing every recognition method with columns for employee count and manager count.
- Below: the **top 3 most common recognition methods** according to employees, and separately according to managers.

### 8.5 Employee Input in the Department (Employee vs Manager)

- Left column: how many employees said Yes/No to "my manager seeks and considers my input". Names of "No" responders are listed.
- Right column: how many managers said Yes/No to "I actively seek my team members' input". Names of subordinates whose manager answered "No" are listed.

### 8.6 Ways to Address Mistakes (Employee vs Manager)

- A grouped bar chart comparing how employees vs. managers describe the approach to mistakes (e.g. constructive feedback, coaching, immediate confrontation, blame approach).
- Below: a **cases to review** section highlighting:
  - Managers who selected **undesired approaches** (Immediate Confrontation or Blame Approach)
  - The specific employees who mentioned those managers as using those approaches

### 8.7 Employee Integration within the Team (Employee vs Manager)

- Two side-by-side panels showing how many employees are **well integrated** according to themselves vs. according to their manager.
- Includes the percentage for each side.

---

## 9. Using the Compare (Year-over-Year) Section

This section lets you compare KPIs across **two different years** (e.g. 2024 vs. 2025).

### Step-by-step

1. Select **Compare** from the sidebar.
2. Open the **Upload** expander.
3. You will see two columns:

| Baseline Period (left) | Comparison Period (right) |
|------------------------|---------------------------|
| Employee Check-In (e.g. 2024) | Employee Check-In (e.g. 2025) |
| Manager Check-In (e.g. 2024) | Manager Check-In (e.g. 2025) |
| Mena Report (e.g. 2024) | Mena Report (e.g. 2025) |

4. Upload **three files per period** (six files total).
5. (Optional) Choose a department filter.
6. Click **Run Comparison**.
7. If errors are found, they appear in a single red block grouped by period (Baseline / Comparison). Fix them and try again.

### Validation rules for Compare

- The **Baseline year must be earlier** than the Comparison year (e.g. 2024 < 2025).
- Within each period, the Employee and Manager check-in timestamps must be from the **same year**.
- The Mena Report **filename** must include a year matching the check-in data year for that period.

### Quick Summary

Once processing succeeds, four delta-indicator cards appear at the top:

| Card | What it shows |
|------|---------------|
| Employee Stress | Stress rate in the comparison year, with change from baseline |
| At Risk Rate | Percentage of at-risk employees, with change |
| Manager Stress | Manager stress rate, with change |
| HR Pulse Requests | Pulse-check request rate, with change |

Green arrows (↓) mean the metric **improved** (e.g. stress went down). Red arrows (↑) mean it **worsened**.

### Comparison views

Use the radio buttons to switch between three tabs:

#### Employee Insights tab

| Section | What it shows |
|---------|---------------|
| Additional Employee KPIs Comparison | A table with counts and percentages for both years, plus the change |
| Stress Analysis | Side-by-side stress frequency breakdowns with delta indicators |
| Pulse-Check Meeting Requests | Yes-count for both years with delta |
| Supportive Work Environment | Yes/No counts for both years with delta |
| Department Dynamics Enhancement | Side-by-side horizontal bar charts |

#### Manager Insights tab

| Section | What it shows |
|---------|---------------|
| Additional Manager KPIs Comparison | A table with counts and percentages for both years, plus the change |
| At Risk of Low Performance | Count and top-10 names for both years with delta |
| Manager Stress about Employees | Stress rate for both years with delta |
| Promotion Readiness | Yes/No counts for both years with delta |
| Company Culture Fit | Fit/Don't fit counts for both years with delta |

#### Combined Analysis tab

| Section | What it shows |
|---------|---------------|
| Changes in Job Responsibilities | Employee vs. Manager "Yes" counts for both years |
| Check-in Meeting Frequency | Side-by-side frequency charts for both years |
| Reward & Recognition Methods | Full comparison table across years and perspectives |

### Important notes

- Each period must use its **own Mena Report** from the corresponding year. The Mena Report changes over time (new hires, departures, name corrections).
- The tool **enforces** that the Baseline year is strictly less than the Comparison year.
- All delta indicators use "inverse" colour logic: increases in negative metrics (stress, risk) show as red, while decreases show as green.

---

## 10. NLP — Open-Ended Answer Analysis

### What is it?

The NLP module uses AI (via GitHub Models) to analyse the **free-text, open-ended answers** in the Employee Check-In. It detects themes, sentiment, severity, and generates actionable recommendations for each response.

### Prerequisites

1. A **GITHUB_MODELS_TOKEN** environment variable must be set before launching the tool. This is a fine-grained GitHub Personal Access Token with the `models:read` permission. Ask your IT team if you don't have one.
2. Employee Check-In data must be **uploaded and cleaned** in the KPIs section first.

### How to use it

1. In the sidebar, make sure **KPIs** is selected.
2. Under "Subsections", check the **NLP** checkbox.
3. Scroll down to the "NLP — Open-Ended Answer Analysis" section.
4. (Optional) Filter by department using the multi-select dropdown.
5. Select which open-ended questions to analyse. By default, all detected questions are selected.
6. Click **Run NLP Analysis**. A progress bar will track the processing.

### Available questions

The NLP module recognises 16 built-in question types from the Employee Check-In:

| # | Question | Conditional? |
|---|----------|-------------|
| 1 | Goal alignment elaboration | Yes — only when employee answered "No" to goal alignment |
| 2 | Professional goals elaboration | Yes — only when employee answered "No" to discussing professional goals |
| 3 | Goals completed | No |
| 4 | Goals in progress | No |
| 5 | Obstacles to pending goals | No |
| 6 | Positive performance behaviors | No |
| 7 | Behaviors developed (past 6 months) | No |
| 8 | Behaviors to develop (next 6 months) | No |
| 9 | Support needed for behavior development | No |
| 10 | Reasons for inability to adapt | Yes — only when adaptation score is 1 or 2 |
| 11 | Growth alignment elaboration | Yes — only when employee answered "No" to growth alignment |
| 12 | Example of input taken into consideration | Yes — only when employee answered "Yes" to input being considered |
| 13 | Collaborative culture elaboration | Yes — only when employee answered "No" to culture question |
| 14 | Ideal work culture description | No |
| 15 | Team detachment elaboration | Yes — only when employee answered "Indifferent" or "Detached" |
| 16 | Recommendation elaboration | Yes — only when employee answered "No" to recommending the company |

### NLP output

After analysis, the following sections appear:

#### Summary KPIs
Four cards: Total records, Substantive answers, Non-answers, and Questions analysed.

#### Sentiment Distribution
A bar chart and table showing how many responses were **positive**, **neutral**, or **negative**.

#### Theme Distribution
A horizontal bar chart of all detected themes. The fixed taxonomy includes:
- Role clarity, Workload, Process/Workflow, Tools/Systems, Training/Enablement, Communication, Manager support, Career growth, Culture/Environment, Work-life balance, Other/Unclear

#### Severity Distribution
A table counting how many responses fall into each severity level:
- **0 = None**, **1 = Low**, **2 = Medium**, **3 = High**

#### Top 5 Issues (Severity × Count)
The five most critical themes, ranked by a combined score of how often they appear and how severe they are.

#### Top Issues by Department
The top 3 themes for each department.

#### Top Issues by Manager
The top 3 themes grouped by the employee's manager.

#### Detailed Results Table
The full record-level data with columns including employee name, department, manager, question, raw answer, detected themes, sentiment, severity, actionability score, a short summary, and a recommendation.

#### Download
CSV and Excel exports of the full NLP results.

#### Audit Sample
A random sample of records for manual spot-checking. Use the slider to adjust how many rows to display (up to 50).

---

## 11. Downloads & Exports

Throughout the tool, you will find download buttons next to tables and charts. Here is a summary:

| Location | What you can download | Format |
|----------|-----------------------|--------|
| Manager Behaviors chart | Venn diagram image | PNG |
| Department Dynamics chart (Employee) | Bar chart image | PNG |
| Department Dynamics chart (Manager) | Bar chart image | PNG |
| Stress Reasons table | Table of reasons and names | CSV |
| At-Risk details | At-risk employees with reasons and PIP status | CSV |
| Ready for promotion table | Promotion-ready employees | CSV |
| Not-ready review dates | Review schedule by month | CSV |
| Unparsed dates diagnostics | Entries with date issues | CSV |
| Job changes types chart | Grouped bar chart | PNG |
| Reward & Recognition table | Methods comparison | CSV |
| Mistakes chart | Grouped bar chart | PNG |
| Check-in frequency chart | Grouped bar chart | PNG |
| Compare — KPI comparison tables | Employee or Manager KPI tables | CSV |
| Compare — dynamics chart | Year-over-year chart | PNG |
| Compare — frequency chart | Year-over-year chart | PNG |
| Compare — recognition table | Multi-year comparison | CSV |
| NLP detailed results | Full analysis data | CSV, Excel |

> **Tip:** PNG downloads capture the chart exactly as displayed in the tool. CSV downloads contain the raw data behind each table.

---

## 12. How Name Matching Works

The tool matches survey respondents to the Mena Report using a two-step process:

### Step 1: Exact Email Match

The full email address from the survey is compared to the Email column in the Mena Report. If a match is found, the canonical **Employee Name** from Mena is used.

### Step 2: Local-Part Fallback

If the full email doesn't match (e.g. different domain), the tool extracts the **local part** (everything before the `@` sign) and tries to match that. This handles cases where the same person has different email domains across systems.

### Match Source column

After matching, each row gets a `Match Source` value:

| Value | Meaning |
|-------|---------|
| **ExactEmail** | The full email matched exactly in the Mena Report |
| **LocalPart** | Only the part before `@` matched — review these for accuracy |
| **Unmatched** | No match found — the original survey name is kept as-is |

> **Important:** The tool **never overwrites** the original survey names. It adds new columns (`Mena Name` for employees, `Name on Mena` for managers) alongside the originals.

---

## 13. Error Messages — What They Mean and How to Fix Them

When errors are found, they appear in a **single red box** grouped by category. Here is a complete reference:

### Missing or Unreadable Files

| Error | Cause | Fix |
|-------|-------|-----|
| *"Employee Check-In file is missing."* | No Employee Check-In was uploaded | Upload the Employee Check-In file |
| *"Manager Check-In file is missing."* | No Manager Check-In was uploaded | Upload the Manager Check-In file |
| *"Mena Report file is missing."* | No Mena Report was uploaded (Compare section) | Upload the Mena Report |
| *"No Mena Report found."* | No upload and no auto-detected file in `Data/` | Upload a Mena Report or place one in the `Data/` folder |
| *"File '...' could not be read — it may be corrupted, password-protected, or in an unsupported format."* | The file cannot be opened | Open it in Excel, confirm it opens, re-save as `.xlsx`, and upload again |
| *"Mena Report (...) could not be read — ..."* | Same as above for the Mena Report | Re-save the Mena Report as `.xlsx` from Excel |
| *"File appears empty or unreadable."* | The file uploaded has no data rows | Check that the file contains data and is not empty |

### Column Errors

| Error | Cause | Fix |
|-------|-------|-----|
| *"Employee Check-In file must contain a column named 'Your Manager's Name'."* | The required column header is missing | Ensure your spreadsheet has a column header exactly matching **"Your Manager's Name"** |
| *"Manager Check-In file must contain a column named 'Subordinate Name'."* | The required column header is missing | Ensure your spreadsheet has a column header exactly matching **"Subordinate Name"** |
| *"Mena Report is missing required columns: ['Email', ...]"* | One or more of the three required columns is missing | Ensure the Mena Report has columns: **Employee Name**, **Email**, **Manager Name** |

### Year Consistency Errors

| Error | Cause | Fix |
|-------|-------|-----|
| *"Year mismatch across uploaded files: Employee Check-In → 2025, Manager Check-In → 2024."* | The timestamps in the two files are from different years | Re-upload the correct year's file so both match |
| *"Could not extract a year from the Timestamp column in '...'."* | The file has no Timestamp column or the dates could not be parsed | Ensure the file has a Timestamp column with valid dates |
| *"Mena Report filename must include a year (e.g. 2025)."* | The Mena Report filename does not contain a 4-digit year | Rename the file to include the year (e.g. `Mena Report 2025.xlsx`) |
| *"Mena Report year (2024) does not match the check-in data year (2025)."* | The year in the Mena filename doesn't match the check-in data | Upload the Mena Report from the correct year |
| *"Baseline year (2025) must be earlier than comparison year (2024)."* | The Baseline period has a later year than the Comparison period | Swap the files so the earlier year is in the Baseline column |

### Processing Errors

| Error | Cause | Fix |
|-------|-------|-----|
| *"Cleaning failed: ..."* | An unexpected issue during data processing | Check the file opens correctly in Excel. If the issue persists, contact IT with the exact error message |

### NLP Errors

| Error | Cause | Fix |
|-------|-------|-----|
| *"GITHUB_MODELS_TOKEN environment variable is not set."* | No API token configured | Ask IT to set the `GITHUB_MODELS_TOKEN` environment variable |
| *"NLP analysis failed: ..."* | An issue with the AI model or network | Check your internet connection and try again. If the issue persists, contact IT |

---

## 14. Frequently Asked Questions

**Q: Can I use the tool with only an Employee Check-In (without the Manager file)?**
A: No. Both the Employee Check-In and Manager Check-In files are required for the tool to process correctly.

**Q: Does the tool modify my original files?**
A: No. The tool reads your files and produces results in the browser. Your original files are never changed.

**Q: What does "Match Source" mean?**
A: See [Section 12](#12-how-name-matching-works) for a full explanation. In short: it tells you whether the name was matched by full email, by the local-part only, or was unmatched.

**Q: Can I filter by department after running?**
A: Yes. Change the department in the dropdown and all KPIs update automatically — you do not need to click Run again.

**Q: Why do I need a separate Mena Report for each year in Compare?**
A: The employee roster changes each year — new hires, departures, and name corrections mean the Mena Report from 2024 may not be accurate for 2025 data.

**Q: What file formats are supported?**
A: `.xlsx` (recommended), `.xls`, and `.csv`.

**Q: Can I upload files with Arabic or special characters in the name?**
A: Yes. The tool handles filenames with any characters. The key requirements are valid data inside the file and correct column headers.

**Q: What if I see "N/A" for a KPI?**
A: This means the tool could not detect the relevant survey question column in your data. This typically happens when the survey format changes between years — the question header may differ slightly. Contact your IT team to review column mappings.

**Q: Can I run the NLP analysis on Manager Check-In data?**
A: Currently, the NLP module only supports Employee Check-In open-ended answers. Manager Check-In NLP support may be added in a future version.

**Q: How long does the NLP analysis take?**
A: It depends on the number of records and questions. For ~50 employees and 16 questions, expect 2–5 minutes. A progress bar tracks the completion.

**Q: What do the delta colours mean in Compare?**
A: For metrics where **lower is better** (stress, at-risk rate), a decrease shows as **green** and an increase shows as **red**. This is the "inverse" colour logic used throughout the comparison section.

**Q: Can I download the cleaned data (before KPIs are calculated)?**
A: The cleaned data is used internally for KPI calculations but is not currently exposed as a separate download. You can download specific KPI tables and charts using the download buttons throughout the tool.

**Q: What if some employees don't match the Mena Report?**
A: They will appear with `Match Source = Unmatched`. Their original survey name is preserved. Consider checking:
- Is the employee's email spelled correctly in the survey?
- Is the employee listed in the Mena Report?
- Is the email domain different? (The local-part fallback may catch this, but verify.)

---

## 15. Glossary

| Term | Definition |
|------|-----------|
| **Mena Report** | The official employee directory used as the single source of truth for names, emails, and manager relationships |
| **Canonical name** | The "official" employee name as recorded in the Mena Report |
| **Employee Check-In** | A periodic survey completed by employees about their goals, stress, work environment, and more |
| **Manager Check-In** | A periodic survey completed by managers evaluating their subordinates on performance risk, promotion readiness, stress, and more |
| **Match Source** | An audit column showing how an employee was matched to the Mena Report: ExactEmail, LocalPart, or Unmatched |
| **KPI** | Key Performance Indicator — a measurable metric derived from the survey data |
| **Baseline Period** | The earlier year in a Year-over-Year comparison (e.g. 2024) |
| **Comparison Period** | The later year in a Year-over-Year comparison (e.g. 2025) |
| **Delta indicator** | The arrow and number showing the change between the Baseline and Comparison period |
| **NLP** | Natural Language Processing — the AI-powered analysis of open-ended text answers |
| **Sentiment** | The emotional tone of a text response: positive, neutral, or negative |
| **Severity** | How critical an issue mentioned in a response is: 0 (none) to 3 (high) |
| **Actionability** | How easy it is to act on the issue mentioned: 0 (low) to 2 (high) |
| **Theme** | A category of topic detected in an open-ended answer (e.g. "Workload", "Manager support", "Career growth") |
| **PIP** | Performance Improvement Plan — a formal development plan for employees at risk of low performance |
| **Local part** | The portion of an email address before the `@` sign (e.g. "john.doe" in john.doe@company.com) |
| **Company Name / Department** | The combined company-and-department field used for filtering; the tool does not split this into separate columns |

---

*Last updated: March 2026*
