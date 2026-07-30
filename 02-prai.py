import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def load_dataset(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return pd.read_csv(path)


def analyze_results(df):
    df = df.copy()
    # show info and missing values
    try:
        print('\nDataFrame info:')
        df.info()
    except Exception:
        pass
    try:
        print('\nMissing values per column:')
        print(df.isnull().sum())
    except Exception:
        pass

    # drop student_id if present
    if 'student_id' in df.columns:
        df = df.drop('student_id', axis=1)
        print('\nDropped column: student_id')

    # print first rows after potential drop
    try:
        print('\nFirst 5 rows after preprocessing:')
        print(df.head())
    except Exception:
        pass
    score_columns = [col for col in df.columns if 'score' in col.lower() or col.lower() in ('math score', 'reading score', 'writing score')]
    if not score_columns:
        score_columns = [col for col in df.columns if col.lower() in ('math score', 'reading score', 'writing score')]

    df['average_score'] = df[score_columns].mean(axis=1)
    df['pass_status'] = df['average_score'].apply(lambda x: 'Pass' if x >= 60 else 'Fail')

    # normalize study hours column if present
    study_cols = [c for c in df.columns if c.replace('_', ' ').lower() in ('study hours', 'study_hours', 'hours studied')]
    if study_cols:
        sc = study_cols[0]
        # coerce to numeric and fill missing with median
        df[sc] = pd.to_numeric(df[sc], errors='coerce')
        median_hours = df[sc].median()
        df[sc] = df[sc].fillna(median_hours)
        print(f"\nFound study hours column: '{sc}'. Filled missing with median: {median_hours}")
        # correlation with average score
        try:
            corr = df[[sc, 'average_score']].corr().iloc[0, 1]
            print(f"Correlation between {sc} and average_score: {corr:.3f}")
        except Exception:
            pass

    print('Dataset shape:', df.shape)
    print('\nColumns:')
    print(df.columns.tolist())

    print('\nFirst 5 rows:')
    print(df.head())

    print('\nSummary statistics for score columns:')
    print(df[score_columns + ['average_score']].describe())

    print('\nPass/Fail counts:')
    print(df['pass_status'].value_counts())

    if 'gender' in df.columns:
        print('\nGender distribution:')
        print(df['gender'].value_counts())
        print('\nGender distribution (percentage):')
        print(df['gender'].value_counts(normalize=True) * 100)
        print('\nAverage scores by gender:')
        print(df.groupby('gender')[score_columns + ['average_score']].mean())

    for col in ['race/ethnicity', 'parental level of education', 'lunch', 'test preparation course']:
        if col in df.columns:
            print(f'\nAverage scores by {col}:')
            print(df.groupby(col)[score_columns + ['average_score']].mean())
            
            # Include additional numerical columns in groupby
            additional_cols = [c for c in df.columns if c.lower() in ('attendance_percent', 'sleep_hours', 'internet_access', 'extraccurricular_activities', 'part_time_job', 'previous_grade')]
            if additional_cols:
                print(f'\nGrouped statistics by {col} (including additional columns):')
                print(df.groupby(col)[score_columns + ['average_score'] + additional_cols].mean())
            # Additional detailed grouping for parental education
            if col == 'parental level of education':
                try:
                    grp = df.groupby(col)
                    # detect attendance and cocurricular columns if present
                    add_cols = []
                    # common names for attendance percent
                    for c in df.columns:
                        if c.replace('_', ' ').lower() in ('attendance', 'attendance percentage', 'attendance_percent', 'attendance_percentage') or 'attendance' in c.lower():
                            add_cols.append(c)
                            break
                    # common names for cocurricular activities
                    coc_cols = []
                    for c in df.columns:
                        if c.replace('_', ' ').lower() in ('cocurricular activities', 'co curricular activities', 'cocurricular', 'co_curricular', 'co_curricular_activities') or 'cocurricular' in c.lower() or 'co-curricular' in c.lower() or 'co curricular' in c.lower():
                            coc_cols.append(c)
                    # build aggregation list
                    agg_cols = score_columns + ['average_score'] + add_cols
                    summary = grp[agg_cols].agg(['count', 'mean', 'median', 'std'])
                    # compute pass rate
                    pass_rate = grp.apply(lambda x: (x['pass_status'] == 'Pass').mean())
                    # compute cocurricular participation rate if such columns exist
                    coc_participation = None
                    if coc_cols:
                        # try to compute participation rate assuming values like Yes/No or bool or counts
                        def participation_rate(series):
                            s = series.dropna()
                            if s.empty:
                                return float('nan')
                            # boolean
                            if s.dtype == bool:
                                return s.mean()
                            # yes/no
                            lower = s.astype(str).str.lower()
                            if set(lower.unique()) <= {'yes', 'no'} or set(lower.unique()) & {'yes'}:
                                return (lower == 'yes').mean()
                            # numeric
                            try:
                                return pd.to_numeric(s, errors='coerce').notna().mean()
                            except Exception:
                                return float('nan')

                        coc_participation = {col: grp[col].apply(participation_rate) for col in coc_cols}
                    print(f"\nDetailed summary by {col}:")
                    print(summary)
                    print(f"\nPass rate by {col}:")
                    print(pass_rate.sort_values(ascending=False))
                    if add_cols:
                        print(f"\nAverage attendance by {col}:")
                        try:
                            print(grp[add_cols].mean())
                        except Exception:
                            pass
                    if coc_participation:
                        print(f"\nCocurricular participation rate by {col}:")
                        try:
                            # print each cocurricular column participation rate
                            for k, v in coc_participation.items():
                                print(f"{k}:\n", v.sort_values(ascending=False))
                        except Exception:
                            pass
                    # plot average score by parental education
                    try:
                        plt.figure(figsize=(10, 5))
                        order = df.groupby(col)['final_score'].mean().sort_values(ascending=False).index
                        sns.barplot(x=grp['final_score'].mean().loc[order].index, y=grp['final_score'].mean().loc[order].values)
                        plt.xticks(rotation=45, ha='right')
                        plt.title('Average Score by Parental Level of Education')
                        plt.ylabel('Average Score')
                        plt.tight_layout()
                        plt.show()
                    except Exception:
                        pass
                    # heatmap of mean scores by parental education
                    try:
                        heatmap_data = grp[score_columns + ['average_score']].mean()
                        plt.figure(figsize=(10, 6))
                        sns.heatmap(heatmap_data.T, annot=True, fmt='.2f', cmap='YlGnBu', cbar_kws={'label': 'Score'})
                        plt.title(f'Heatmap of Mean Scores by {col}')
                        plt.xlabel(col)
                        plt.ylabel('Score Columns')
                        plt.tight_layout()
                        plt.show()
                    except Exception:
                        pass
                except Exception:
                    pass

    print('\nTop 10 students by average score:')
    if 'student_name' in df.columns:
        print(df.sort_values('average_score', ascending=False).head(10)[['student_name', 'average_score'] + score_columns])
    else:
        print(df.sort_values('average_score', ascending=False).head(10)[score_columns + ['average_score']])

    # Visualizations using seaborn
    try:
        plt.figure(figsize=(8, 6))
        sns.histplot(df['average_score'], kde=True, bins=20)
        plt.title('Distribution of Average Scores')
        plt.xlabel('Average Score')
        plt.ylabel('Count')
        plt.tight_layout()
        plt.show()

        if 'gender' in df.columns:
            plt.figure(figsize=(4, 3))
            sns.countplot(x='gender', data=df)
            plt.title('Gender Distribution')
            plt.xlabel('Gender')
            plt.ylabel('Count')
            plt.tight_layout()
            plt.show()

        if 'gender' in df.columns:
            plt.figure(figsize=(8, 6))
            sns.boxplot(x='gender', y='average_score', data=df)
            plt.title('Average Score by Gender')
            plt.tight_layout()
            plt.show()

        if len(score_columns) >= 2:
            plt.figure(figsize=(8, 6))
            sns.pairplot(df[score_columns + ['average_score']].dropna())
            plt.suptitle('Pairplot of Score Columns', y=1.02)
            plt.show()
    except Exception:
        # If plotting fails, continue without stopping analysis
        pass

    return df


def main():
    dataset_path = Path(r'C:\Users\anush\Downloads\student_performance_dataset.csv')
    try:
        df = load_dataset(dataset_path)
    except FileNotFoundError as error:
        print(error)
        return

    analyze_results(df)


if __name__ == '__main__':
    main()

 