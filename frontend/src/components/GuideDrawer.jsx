import React, { useState, useEffect } from 'react';

const LatexMath = ({ math, block = false }) => {
  const ref = React.useRef(null);
  useEffect(() => {
    if (ref.current && window.katex) {
      try {
        window.katex.render(math, ref.current, {
          displayMode: block,
          throwOnError: false
        });
      } catch (err) {
        console.error("KaTeX error:", err);
      }
    }
  }, [math, block]);
  return <span ref={ref}>{math}</span>;
};

const renderLinks = (links) => (
  <div style={{ marginTop: '1.5rem', paddingTop: '1rem', borderTop: '1px dashed var(--border-color)' }}>
    <span style={{ fontSize: '0.75rem', fontWeight: 700, color: 'var(--text-main)', display: 'block', marginBottom: '0.5rem', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
      <i className="fa-solid fa-link" style={{ marginRight: '0.35rem', color: 'var(--accent-purple)' }}></i>
      Learn More & References
    </span>
    <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.35rem', margin: 0 }}>
      {links.map((link, idx) => (
        <li key={idx} style={{ fontSize: '0.8rem' }}>
          <a href={link.url} target="_blank" rel="noopener noreferrer" style={{ color: 'var(--accent-green)', textDecoration: 'underline', transition: 'color 0.15s ease' }} onMouseEnter={(e) => e.target.style.color = 'var(--text-main)'} onMouseLeave={(e) => e.target.style.color = 'var(--accent-green)'}>
            {link.text}
          </a>
        </li>
      ))}
    </ul>
  </div>
);

const guideTopics = {
  data_upload: {
    title: 'Data Upload & Ingest',
    concept: (
      <div>
        <p><strong>What is this step?</strong></p>
        <p>In machine learning, the first step is uploading your dataset (e.g. CSV, Excel, or Parquet). The application parses the dataset and automatically predicts the target column (the variable you want to predict) and decides if the task is **Classification** (predicting a discrete category) or **Regression** (predicting a continuous value).</p>
        
        <p style={{ marginTop: '1rem' }}><strong>When to choose Classification vs Regression?</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <li><strong>Classification</strong>: When the target column consists of finite labels or classes. Examples: predicting if an email is spam (Spam / Not Spam), or predicting customer churn (Yes / No).</li>
          <li><strong>Regression</strong>: When the target is a continuous numerical quantity. Examples: predicting house prices ($200k, $450k, etc.) or stock market indices.</li>
        </ul>
        
        <p style={{ marginTop: '1rem' }}><strong>Heuristics & Target Auto-Detection:</strong></p>
        <p>The system determines the task automatically by checking the cardinality (number of unique values) of the target column. If the unique values are few (e.g., less than 20 or less than 5% of the total dataset size), the target is assumed to represent categories (Classification). Otherwise, it is inferred as continuous numerical values (Regression).</p>
        {renderLinks([
          { text: "Scikit-Learn Preprocessing Guide", url: "https://scikit-learn.org/stable/modules/preprocessing.html" },
          { text: "Introduction to Machine Learning Tasks (Wikipedia)", url: "https://en.wikipedia.org/wiki/Machine_learning" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>Target Variable Formats:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <li><strong>Classification:</strong> The target set is discrete:
            <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
              <LatexMath math="Y \in \{ C_1, C_2, \dots, C_k \}" block />
            </div>
            where each <LatexMath math="C_i" /> represents a unique class label.
          </li>
          <li><strong>Regression:</strong> The target set is continuous real numbers:
            <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
              <LatexMath math="Y \in \mathbb{R}" block />
            </div>
          </li>
        </ul>
        
        <p style={{ marginTop: '1.25rem' }}><strong>Auto-Detection Criterion (Cardinality Ratio):</strong></p>
        <p>Let <LatexMath math="N" /> be the total number of rows in the dataset, and let <LatexMath math="U_y" /> be the number of unique values in the target column <LatexMath math="y" />. The cardinality ratio <LatexMath math="R" /> is defined as:</p>
        <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '6px', margin: '0.75rem 0' }}>
          <LatexMath math="R = \frac{U_y}{N}" block />
        </div>
        <p>The system predicts the task based on the following rule:</p>
        <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '6px', margin: '0.75rem 0', fontSize: '0.8rem', lineHeight: '1.5' }}>
          <strong>Task Selection Rule:</strong><br />
          IF <LatexMath math="U_y \le 2" /> OR (<LatexMath math="U_y \le 20" /> AND <LatexMath math="R < 0.05" />) → <strong>Classification</strong><br />
          ELSE → <strong>Regression</strong>
        </div>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation:</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.8rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`import pandas as pd

def predict_target_and_task(df: pd.DataFrame):
    """
    Predict which column is the target and determine the task
    (Classification or Regression) based on cardinality.
    """
    # 1. We assume the last column is the default target
    target_col = df.columns[-1]
    
    # 2. Compute uniqueness metrics
    n_rows = len(df)
    unique_vals = df[target_col].nunique()
    cardinality_ratio = unique_vals / n_rows
    
    # 3. Check data types and apply rules
    dtype = df[target_col].dtype
    is_numeric = pd.api.types.is_numeric_dtype(df[target_col])
    
    if not is_numeric or unique_vals <= 2 or (unique_vals <= 20 and cardinality_ratio < 0.05):
        task = "Classification"
    else:
        task = "Regression"
        
    return target_col, task`}
        </pre>
        <p style={{ marginTop: '1rem' }}><strong>Code Explanation:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.85rem' }}>
          <li>Line 9: Assigns the last column of the DataFrame as the target by default.</li>
          <li>Line 12-13: Computes dataset size and the cardinality ratio.</li>
          <li>Line 18: If the target column is non-numeric (strings/objects) or has very low cardinality, it forces **Classification**, as models cannot perform regression on categories.</li>
        </ul>
      </div>
    )
  },
  duplicates: {
    title: 'Duplicate Row Filtering',
    concept: (
      <div>
        <p><strong>What are duplicate rows?</strong></p>
        <p>Duplicate rows are rows that contain identical values across all columns. In statistical modeling, duplicate rows can severely bias model metrics.</p>
        
        <p style={{ marginTop: '1rem' }}><strong>Why remove duplicates?</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <li><strong>Avoid Overfitting:</strong> A model might memorize duplicated rows, achieving high training accuracy while failing to generalize to unseen data.</li>
          <li><strong>Prevent Data Leakage:</strong> If duplicates exist, the same row might end up in both the training set and the validation set. This artificially inflates evaluation metrics (e.g. accuracy, F1-score).</li>
          <li><strong>Save Computational Power:</strong> De-duplicating speeds up preprocessing and training.</li>
        </ul>
        {renderLinks([
          { text: "Pandas drop_duplicates API Reference", url: "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.drop_duplicates.html" },
          { text: "Handling Duplicate Data in Python (Towards Data Science)", url: "https://towardsdatascience.com/finding-and-reducing-duplicate-data-in-python-6415f187a701" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>Duplicate Definition:</strong></p>
        <p>Let a dataset be represented as a set of rows <LatexMath math="X = \{ r_1, r_2, \dots, r_N \}" />, where each row <LatexMath math="r_i" /> is a vector of features <LatexMath math="[f_{i1}, f_{i2}, \dots, f_{iM}]" />.</p>
        <p>Two distinct rows <LatexMath math="r_i" /> and <LatexMath math="r_j" /> (where <LatexMath math="i \neq j" />) are duplicates if and only if:</p>
        <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '6px', margin: '0.75rem 0' }}>
          <LatexMath math="r_i[k] = r_j[k] \quad \forall k \in \{1, 2, \dots, M\}" block />
        </div>
        <p>When dropping duplicates, we retain the first occurrence <LatexMath math="r_i" /> and discard the subsequent occurrences <LatexMath math="r_j" />, leaving only unique vectors in our feature space.</p>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation:</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.8rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`# Pandas implementation to remove duplicate rows
# Keeps the first occurrence and drops all others
df_cleaned = df.drop_duplicates()`}
        </pre>
        <p style={{ marginTop: '1rem' }}><strong>Code Explanation:</strong></p>
        <p>Pandas internally hashes each row vector. `drop_duplicates()` checks row hashes and returns a copy of the DataFrame with duplicate rows deleted. It operates in O(N) time using hashing.</p>
      </div>
    )
  },
  imputation: {
    title: 'Missing Value Imputation',
    concept: (
      <div>
        <p><strong>What is Imputation?</strong></p>
        <p>Imputation is the process of replacing missing values (NaN or null) in a column with substituted values. Most machine learning algorithms (like XGBoost, Random Forests, or SVMs) cannot handle missing values out of the box and will crash during training.</p>
        
        <p style={{ marginTop: '1rem' }}><strong>Common Imputation Strategies:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <li><strong>Mean (Average)</strong>: Replaces missing values with the average of the non-missing values. Best for normally distributed numerical data without heavy outliers.</li>
          <li><strong>Median (Middle Value)</strong>: Replaces missing values with the middle value of sorted data. Best for skewed numerical data (highly robust to outliers).</li>
          <li><strong>Most Frequent (Mode)</strong>: Replaces missing values with the most common value. Ideal for categorical features (e.g. state names, gender).</li>
          <li><strong>Constant Value</strong>: Fills missing cells with a fixed value like `0` or `"Missing"`. Good when missingness itself represents a specific category or indicator.</li>
          <li><strong>Drop Rows</strong>: Discards any rows containing missing values. Only use when missing values represent &lt;5% of the dataset, or when imputation is not logical.</li>
        </ul>
        {renderLinks([
          { text: "Scikit-Learn Imputation of Missing Values", url: "https://scikit-learn.org/stable/modules/impute.html" },
          { text: "How to Handle Missing Data (Towards Data Science)", url: "https://towardsdatascience.com/how-to-handle-missing-data-8646b18db0d4" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>Mathematical Formulations:</strong></p>
        <p>Let <LatexMath math="x = [x_1, x_2, \dots, x_n]" /> be the vector of non-missing values in a column.</p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <li><strong>Mean Imputation value (<LatexMath math="\mu" />):</strong>
            <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
              <LatexMath math="\mu = \frac{1}{n} \sum_{i=1}^n x_i" block />
            </div>
          </li>
          <li><strong>Median Imputation value (<LatexMath math="M" />):</strong>
            Sort the non-missing array: <LatexMath math="x_{sorted} = [x^{(1)}, x^{(2)}, \dots, x^{(n)}]" />.
            <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
              <LatexMath math="M = x^{(\frac{n+1}{2})} \quad \text{if } n \text{ is odd}" block />
              <LatexMath math="M = \frac{x^{(\frac{n}{2})} + x^{(\frac{n}{2} + 1)}}{2} \quad \text{if } n \text{ is even}" block />
            </div>
          </li>
          <li><strong>Mode Imputation value (<LatexMath math="M_o" />):</strong>
            <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
              <LatexMath math="M_o = \text{argmax}_v (\text{Frequency}(v))" block />
            </div>
          </li>
        </ul>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation:</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.8rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`from sklearn.impute import SimpleImputer
import pandas as pd

def impute_column(df: pd.DataFrame, column: str, strategy: str):
    """
    Impute missing values using scikit-learn's SimpleImputer.
    strategy can be: 'mean', 'median', 'most_frequent', 'constant'
    """
    if strategy == "drop":
        return df.dropna(subset=[column])
        
    imputer = SimpleImputer(strategy=strategy)
    # Fit and transform requires 2D shape, so we pass df[[column]]
    df[column] = imputer.fit_transform(df[[column]])
    return df`}
        </pre>
        <p style={{ marginTop: '1rem' }}><strong>Code Explanation:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.85rem' }}>
          <li>Line 11: If strategy is "drop", we drop the rows having NaNs in that specific column.</li>
          <li>Line 13: Initialises `SimpleImputer` with the chosen strategy.</li>
          <li>Line 15: `fit_transform` calculates the impute metric (mean/median/mode) from non-missing cells and applies it to fill missing cells.</li>
        </ul>
      </div>
    )
  },
  scaling: {
    title: 'Feature Scaling',
    concept: (
      <div>
        <p><strong>What is Feature Scaling?</strong></p>
        <p>Scaling standardizes the range of independent variables (features) of data. In raw datasets, features can have vastly different scales (e.g. Age ranging 0–100, Income ranging 10,000–500,000).</p>
        
        <p style={{ marginTop: '1rem' }}><strong>Why is Scaling important?</strong></p>
        <p>Algorithms that compute distances between points (e.g. K-Means clustering, SVMs, t-SNE, KNN) or use gradient descent optimization (Neural Networks, Logistic Regression) are heavily dominated by columns with larger values. Scaling ensures all columns contribute equally.</p>
        
        <p style={{ marginTop: '1rem' }}><strong>Supported Scaling Methods:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <li><strong>StandardScaler (Standardization)</strong>: Shifts data to have a mean of 0 and standard deviation of 1. Keeps shape distributions but changes ranges. Use when features follow a Gaussian/normal distribution.</li>
          <li><strong>MinMaxScaler (Normalization)</strong>: Scales features to a strict range between 0 and 1. Highly sensitive to outliers. Use when features do not follow normal distributions (e.g. image pixels).</li>
          <li><strong>RobustScaler</strong>: Scales features using median and Interquartile Range (IQR). Outliers are ignored during scaling calculations, making it highly robust.</li>
          <li><strong>MaxAbsScaler</strong>: Scales features to [-1, 1] range by dividing by the maximum absolute value. Ideal for sparse datasets containing zeros.</li>
        </ul>
        {renderLinks([
          { text: "Scikit-Learn Preprocessing & Scaling", url: "https://scikit-learn.org/stable/modules/preprocessing.html" },
          { text: "Feature Scaling on Wikipedia", url: "https://en.wikipedia.org/wiki/Feature_scaling" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>Scaling Formulas:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <li><strong>StandardScaler:</strong>
            <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
              <LatexMath math="z = \frac{x - \mu}{\sigma}" block />
            </div>
            where <LatexMath math="\mu" /> is the mean and <LatexMath math="\sigma" /> is the standard deviation.
          </li>
          <li><strong>MinMaxScaler:</strong>
            <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
              <LatexMath math="x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}" block />
            </div>
          </li>
          <li><strong>RobustScaler:</strong>
            <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
              <LatexMath math="x_{scaled} = \frac{x - Q_2(x)}{Q_3(x) - Q_1(x)}" block />
            </div>
            where <LatexMath math="Q_2" /> is the median, and <LatexMath math="Q_3 - Q_1" /> is the Interquartile Range (IQR).
          </li>
        </ul>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation:</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.8rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import pandas as pd

def scale_columns(df: pd.DataFrame, columns: list, method: str):
    """
    Scale numerical columns using Scikit-Learn scalers.
    method can be: 'standard', 'minmax', 'robust'
    """
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
        
    df[columns] = scaler.fit_transform(df[columns])
    return df`}
        </pre>
        <p style={{ marginTop: '1rem' }}><strong>Code Explanation:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.85rem' }}>
          <li>Lines 9-14: Instantiates the selected scaler.</li>
          <li>Line 16: `fit_transform` computes the fit parameters (e.g. mean, standard dev, min, max) on the columns, scales the data, and writes it back to the DataFrame.</li>
        </ul>
      </div>
    )
  },
  encoding: {
    title: 'Categorical Encoding',
    concept: (
      <div>
        <p><strong>What is Categorical Encoding?</strong></p>
        <p>Categorical Encoding is the process of converting category values (like strings or classes) into numerical representations that machine learning models can read.</p>
        
        <p style={{ marginTop: '1rem' }}><strong>Supported Encoding Methods:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <li><strong>One-Hot Encoding</strong>: Creates a new binary indicator column (0 or 1) for each unique category in the feature. Use for nominal categories with no order (e.g. `Color: [Red, Green, Blue]`). Best when cardinality is low (&lt;15 categories).</li>
          <li><strong>Label Encoding</strong>: Assigns an integer value ($0, 1, 2, \dots$) to each category. Use for ordinal categories with an inherent order (e.g. `Education: [HighSchool=0, Bachelors=1, PhD=2]`), or for tree-based algorithms (Random Forest/XGBoost) which process integers efficiently.</li>
        </ul>
        {renderLinks([
          { text: "Scikit-Learn Encoding Categorical Features", url: "https://scikit-learn.org/stable/modules/preprocessing.html#encoding-categorical-features" },
          { text: "One-Hot vs. Label Encoding (Analytics Vidhya)", url: "https://www.analyticsvidhya.com/blog/2020/03/one-hot-encoding-vs-label-encoding-using-scikit-learn/" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>Encoding Mappings:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
          <li><strong>Label Encoding:</strong>
            Maps a set of categorical labels <LatexMath math="C = \{ c_1, c_2, \dots, c_k \}" /> to integers:
            <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
              <LatexMath math="f(c_i) = i - 1" block />
            </div>
            Example: <LatexMath math="\{\text{Red, Green, Blue}\} \to \{0, 1, 2\}" />.
          </li>
          <li><strong>One-Hot Encoding:</strong>
            Maps each category <LatexMath math="c_i" /> to a binary vector of length <LatexMath math="k" />:
            <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
              <LatexMath math="f(c_i) = [0, \dots, 1, \dots, 0] \quad (\text{1 at position } i)" block />
            </div>
            Example: Red → [1, 0, 0], Green → [0, 1, 0], Blue → [0, 0, 1].
          </li>
        </ul>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation:</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.8rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`import pandas as pd
from sklearn.preprocessing import LabelEncoder

def encode_columns(df: pd.DataFrame, columns: list, method: str):
    """
    Encode categorical columns.
    method can be: 'label' or 'onehot'
    """
    if method == "label":
        for col in columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    elif method == "onehot":
        df = pd.get_dummies(df, columns=columns)
        
    return df`}
        </pre>
        <p style={{ marginTop: '1rem' }}><strong>Code Explanation:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.85rem' }}>
          <li>Line 11: Uses Scikit-Learn's `LabelEncoder` to fit categories and overwrite the column with numerical integers.</li>
          <li>Line 13: `pd.get_dummies` automatically expands the specified columns into separate binary indicator columns (e.g. `col_category_name`).</li>
        </ul>
      </div>
    )
  },
  outliers: {
    title: 'Outlier Filtering',
    concept: (
      <div>
        <p><strong>What is an Outlier?</strong></p>
        <p>An outlier is a data point that differs significantly from other observations. Outliers can be caused by measurement errors, sensor glitches, or represent true extreme anomalies.</p>
        
        <p style={{ marginTop: '1rem' }}><strong>Outlier Detection Methods:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <li><strong>IQR Rule (Interquartile Range)</strong>: Cleans columns individually. Simple, transparent, and standard for boxplots. Good for univariate normal or skewed distributions.</li>
          <li><strong>Z-Score Method</strong>: Identifies outliers as points more than a specific number of standard deviations (usually 3.0) away from the mean. Best for strictly Gaussian distributions.</li>
          <li><strong>Isolation Forest</strong>: An unsupervised ensemble algorithm that isolates anomalies by randomly partitioning feature paths. Excellent for multi-dimensional datasets with complex joint outliers.</li>
          <li><strong>Local Outlier Factor (LOF)</strong>: A density-based algorithm comparing a point's local density to its neighbors. Best when clusters have varying densities.</li>
        </ul>
        {renderLinks([
          { text: "Scikit-Learn Novelty and Outlier Detection", url: "https://scikit-learn.org/stable/modules/outlier_detection.html" },
          { text: "Interquartile Range (IQR) on Wikipedia", url: "https://en.wikipedia.org/wiki/Interquartile_range" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>Mathematical Rules:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <li><strong>IQR Method:</strong>
            Let <LatexMath math="\text{IQR} = Q_3 - Q_1" />. A point <LatexMath math="x" /> is an outlier if:
            <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
              <LatexMath math="x < Q_1 - 1.5 \times \text{IQR} \quad \text{or} \quad x > Q_3 + 1.5 \times \text{IQR}" block />
            </div>
          </li>
          <li><strong>Z-Score Method:</strong>
            Standardize the point: <LatexMath math="z = \frac{x - \mu}{\sigma}" />. The point is an outlier if:
            <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
              <LatexMath math="|z| > \theta \quad (\text{usually } \theta = 3.0)" block />
            </div>
          </li>
        </ul>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation (Isolation Forest):</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.8rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`from sklearn.ensemble import IsolationForest
import pandas as pd

def remove_outliers_isolation_forest(df: pd.DataFrame, columns: list, contamination: float = 0.05):
    """
    Remove outliers multivariately using Isolation Forest.
    contamination is the expected proportion of outliers (default 5%).
    """
    # Select numeric columns for the model
    X = df[columns].dropna()
    
    iso = IsolationForest(contamination=contamination, random_state=42)
    # Fit the model and predict (-1 represents outliers, 1 represents inliers)
    preds = iso.fit_predict(X)
    
    inliers = X.index[preds == 1]
    return df.loc[inliers]`}
        </pre>
        <p style={{ marginTop: '1rem' }}><strong>Code Explanation:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.85rem' }}>
          <li>Line 12: Fits the `IsolationForest` on target numerical features.</li>
          <li>Line 14: Predicts class arrays. `-1` denotes rows that isolated quickly (representing outliers), while `1` denotes inliers.</li>
          <li>Line 17: Retains only inlier indexes in the returned dataset version.</li>
        </ul>
      </div>
    )
  },
  pca: {
    title: 'PCA Dimensionality Reduction',
    concept: (
      <div>
        <p><strong>What is Principal Component Analysis (PCA)?</strong></p>
        <p>PCA is an unsupervised linear dimensionality reduction technique. It projects high-dimensional numerical columns onto a smaller set of orthogonal, uncorrelated variables called **Principal Components** (PC1, PC2, etc.), while retaining as much of the original dataset variance as possible.</p>
        
        <p style={{ marginTop: '1rem' }}><strong>When to apply PCA?</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <li><strong>Reduce Collinearity</strong>: Combines highly correlated numeric columns.</li>
          <li><strong>Compress Dataset</strong>: Reduces the feature count when you have dozens of numerical columns, speeding up training.</li>
          <li><strong>Overcoming Curse of Dimensionality</strong>: Helps models generalize better by filtering noise.</li>
        </ul>
        {renderLinks([
          { text: "Scikit-Learn Principal Component Analysis Documentation", url: "https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html" },
          { text: "Visual Introduction to PCA (Setosa.io)", url: "http://setosa.io/ev/principal-component-analysis/" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>Mathematical Formulation:</strong></p>
        <p>1. Center and standardize the data matrix <LatexMath math="X" /> (mean = 0, std = 1).</p>
        <p>2. Calculate the Covariance Matrix <LatexMath math="\Sigma" />:</p>
        <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
          <LatexMath math="\Sigma = \frac{1}{N} X^T X" block />
        </div>
        <p>3. Find Eigenvectors (<LatexMath math="v" />) and Eigenvalues (<LatexMath math="\lambda" />) by solving:</p>
        <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
          <LatexMath math="\Sigma v = \lambda v" block />
        </div>
        <p>Eigenvectors correspond to the directions of the principal components (loading coefficients), and eigenvalues correspond to the variance explained along those directions. We sort eigenvectors by eigenvalues descending, choose top <LatexMath math="k" />, and project:</p>
        <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
          <LatexMath math="T = X W_k" block />
        </div>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation:</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.8rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd

def apply_pca(df: pd.DataFrame, columns: list, n_components: int = 2):
    # Centering and scaling is required before PCA
    X = StandardScaler().fit_transform(df[columns])
    
    # Fit PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X)
    
    # Drop original columns and attach principal components
    df_reduced = df.drop(columns=columns)
    for i in range(n_components):
        df_reduced[f"PC{i+1}"] = pca_result[:, i]
        
    return df_reduced`}
        </pre>
        <p style={{ marginTop: '1rem' }}><strong>Code Explanation:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem', fontSize: '0.85rem' }}>
          <li>Line 7: Standardizes data. Center and scaling is critical so features with naturally larger values don't dominate the variance equations.</li>
          <li>Line 10-11: Instantiates `PCA` and projects the scaled matrix onto the top components.</li>
        </ul>
      </div>
    )
  },
  eda_univariate: {
    title: 'Univariate Analysis',
    concept: (
      <div>
        <p><strong>What is Univariate Analysis?</strong></p>
        <p>Univariate analysis is the simplest form of analyzing data where we inspect a single variable (column) at a time to examine its distribution, center, range, and spread.</p>
        
        <p style={{ marginTop: '1rem' }}><strong>Why inspect variables individually?</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <li><strong>Check skewness</strong>: Identify if a numerical column is skewed or normal. highly skewed target columns might require log-transformations.</li>
          <li><strong>Class Imbalance</strong>: Check if category labels are balanced in classification tasks. Highly imbalanced target columns (e.g. 99% Class A, 1% Class B) require stratified sampling.</li>
          <li><strong>Outlier Identification</strong>: Visualizing spread via box plots helps detect anomalous tails.</li>
        </ul>
        {renderLinks([
          { text: "Exploratory Data Analysis Guide (Wikipedia)", url: "https://en.wikipedia.org/wiki/Exploratory_data_analysis" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>Histogram Frequency Binning:</strong></p>
        <p>For a numeric feature, we divide its values range <LatexMath math="[x_{min}, x_{max}]" /> into <LatexMath math="k" /> equally spaced intervals (bins) of width <LatexMath math="w" />:</p>
        <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
          <LatexMath math="w = \frac{x_{max} - x_{min}}{k}" block />
        </div>
        <p>For each bin interval <LatexMath math="[B_j, B_j + w)" />, the count is computed as the number of data points falling inside the range:</p>
        <div style={{ padding: '0.5rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '4px', margin: '0.5rem 0' }}>
          <LatexMath math="\text{Count}_j = \sum I(x_i \in [B_j, B_j + w))" block />
        </div>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation:</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.8rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`import numpy as np

def compute_histogram(series, bins=30):
    """Computes histogram bins and frequency counts."""
    data = series.dropna()
    counts, bin_edges = np.histogram(data, bins=bins)
    
    bin_labels = []
    for i in range(len(counts)):
        bin_labels.append(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}")
        
    return {
        "labels": bin_labels,
        "counts": counts.tolist(),
        "mean": float(data.mean()),
        "median": float(data.median()),
        "std": float(data.std())
    }`}
        </pre>
      </div>
    )
  },
  eda_bivariate: {
    title: 'Bivariate Analysis',
    concept: (
      <div>
        <p><strong>What is Bivariate Analysis?</strong></p>
        <p>Bivariate analysis involves analyzing the relationship between exactly two columns. A scatter plot is the standard tool to inspect how two numerical features interact, helping you look for linear or non-linear correlation patterns.</p>
        {renderLinks([
          { text: "Bivariate Analysis Detailed Overview", url: "https://en.wikipedia.org/wiki/Bivariate_analysis" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>Scatter Coordinates Mapping:</strong></p>
        <p>Points are mapped as coordinate vectors <LatexMath math="(x_i, y_i)" /> in a 2D Euclidean coordinate space. This is used to visually inspect correlation direction (positive, negative, or independent) and structure (linear, polynomial, clusters, or random noise).</p>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation:</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.8rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`def get_scatter_coordinates(df, col_x, col_y, max_points=1000):
    # Sample to prevent browser lag with large files
    sampled = df[[col_x, col_y]].dropna()
    if len(sampled) > max_points:
        sampled = sampled.sample(n=max_points, random_state=42)
        
    points = [{"x": float(row[col_x]), "y": float(row[col_y])} for _, row in sampled.iterrows()]
    return points`}
        </pre>
      </div>
    )
  },
  eda_correlation: {
    title: 'Correlation Heatmaps',
    concept: (
      <div>
        <p><strong>What is a Correlation Matrix?</strong></p>
        <p>A correlation matrix displays the correlation coefficients between all numerical feature combinations in your dataset. Coefficients range strictly between -1.0 and +1.0.</p>
        
        <p style={{ marginTop: '1rem' }}><strong>Interpreting Correlation Coefficients:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <li><strong>+1.00 (Perfect Positive Correlation)</strong>: As feature X increases, feature Y increases proportionally.</li>
          <li><strong>0.00 (No Correlation)</strong>: Feature X and feature Y have no linear relationship.</li>
          <li><strong>-1.00 (Perfect Negative Correlation)</strong>: As feature X increases, feature Y decreases proportionally.</li>
        </ul>
        
        <p style={{ marginTop: '1rem' }}><strong>Why check correlations?</strong></p>
        <p>Identifying highly correlated features (multicollinearity, e.g. r &gt; 0.85) allows you to drop redundant variables. This prevents model instabilities and over-parameterization, especially in linear models.</p>
        {renderLinks([
          { text: "Pearson Correlation Coefficient on Wikipedia", url: "https://en.wikipedia.org/wiki/Pearson_correlation_coefficient" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>Pearson Correlation Coefficient (r):</strong></p>
        <p>For two variable vectors <LatexMath math="X" /> and <LatexMath math="Y" />, Pearson's <LatexMath math="r" /> is computed as:</p>
        <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '6px', margin: '0.75rem 0' }}>
          <LatexMath math="r = \frac{\sum (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum (x_i - \bar{x})^2 \sum (y_i - \bar{y})^2}}" block />
        </div>
        <p>where <LatexMath math="\bar{x}" /> and <LatexMath math="\bar{y}" /> represent the average values of the columns <LatexMath math="X" /> and <LatexMath math="Y" /> respectively.</p>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation:</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.8rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`import pandas as pd

def compute_correlation(df: pd.DataFrame):
    # Select only numeric features
    numeric_df = df.select_dtypes(include=["number"])
    # Calculate Pearson correlation matrix
    corr = numeric_df.corr().fillna(0)
    
    return {
        "columns": numeric_df.columns.tolist(),
        "matrix": corr.values.tolist()
    }`}
        </pre>
      </div>
    )
  },
  eda_projections: {
    title: 'Dimensional Projections',
    concept: (
      <div>
        <p><strong>What are Dimensional Projections?</strong></p>
        <p>When working with many numerical columns, it's impossible to visualize the dataset shape beyond 3 dimensions. Projection algorithms project multi-dimensional feature rows down to a 2D space coordinate (Dimension 1, Dimension 2) so that we can visually inspect groupings, separations, or manifolds.</p>
        
        <p style={{ marginTop: '1rem' }}><strong>Available Methods:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <li><strong>PCA</strong>: Fast, linear projection that preserves the directions of maximum global variance.</li>
          <li><strong>t-SNE</strong>: Non-linear projection that preserves local structures. Excellent for visual cluster clustering and separating category groupings in 2D space.</li>
          <li><strong>LDA (Linear Discriminant Analysis)</strong>: Supervised projection method that finds directions separating classes the most.</li>
          <li><strong>UMAP</strong>: Preserves both local and global features, running faster than t-SNE.</li>
        </ul>
        {renderLinks([
          { text: "t-SNE Visualizations Explained (Distill.pub)", url: "https://distill.pub/2016/misread-tsne/" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>t-SNE Divergence Minimization:</strong></p>
        <p>t-SNE computes conditional probabilities that represent similarities between points in high-dimensional space (<LatexMath math="p_{ij}" />) and low-dimensional space (<LatexMath math="q_{ij}" />). It then minimizes the Kullback-Leibler (KL) divergence between these two probability distributions:</p>
        <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '6px', margin: '0.75rem 0' }}>
          <LatexMath math="\text{KL}(P \parallel Q) = \sum_i \sum_j p_{ij} \log\left(\frac{p_{ij}}{q_{ij}}\right)" block />
        </div>
        <p>This is optimized using gradient descent to find the optimal 2D coordinates.</p>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation:</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.8rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import pandas as pd

def compute_tsne_2d(df: pd.DataFrame, numeric_cols: list):
    # Scale variables
    X = StandardScaler().fit_transform(df[numeric_cols].fillna(0))
    
    # Initialize t-SNE (perplexity represents neighbor density search)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    reduced = tsne.fit_transform(X)
    
    return [{"x": float(reduced[i, 0]), "y": float(reduced[i, 1])} for i in range(len(reduced))]`}
        </pre>
      </div>
    )
  },
  training: {
    title: 'Model Training & Tuning',
    concept: (
      <div>
        <p><strong>What happens in this step?</strong></p>
        <p>Here we train classification or regression models. We support algorithms like Random Forests, Gradient Boosting (XGBoost/LightGBM), Support Vector Machines, and Linear Classifiers. The system splits the data, tunes hyperparameters, and validates the model.</p>
        
        <p style={{ marginTop: '1rem' }}><strong>Tuning, Underfitting, and Overfitting:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
          <li><strong>Overfitting</strong>: The model memorizes training noise and fails on unseen testing data. Avoided using cross-validation and regularisation.</li>
          <li><strong>Underfitting</strong>: The model is too simple to capture patterns. Solved by hyperparameter tuning and model capacity increases.</li>
          <li><strong>Optuna Hyperparameter Tuning</strong>: Uses Tree-structured Parzen Estimators (TPE) to search for optimal parameter settings (e.g. learning rate, number of estimators, max depth) automatically.</li>
        </ul>
        {renderLinks([
          { text: "Scikit-Learn Tuning Hyperparameters Guide", url: "https://scikit-learn.org/stable/modules/grid_search.html" },
          { text: "Optuna Official Documentation", url: "https://optuna.org/" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>K-Fold Cross-Validation:</strong></p>
        <p>To prevent data leakage and evaluate generalization, the dataset is divided into <LatexMath math="K" /> equal parts (folds). The model is trained on <LatexMath math="K-1" /> folds and validated on the remaining fold. This is repeated <LatexMath math="K" /> times:</p>
        <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '6px', margin: '0.75rem 0' }}>
          <LatexMath math="\text{CV\_Score} = \frac{1}{K} \sum_{k=1}^K \text{Validation\_Score}_k" block />
        </div>
        <p>Optuna searches for hyperparameters <LatexMath math="\theta" /> that maximize this CV_Score.</p>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation (Optuna Tuning):</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.75rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

def tune_random_forest(X, y):
    def objective(trial):
        # Define hyperparameter search spaces
        n_estimators = trial.suggest_int("n_estimators", 10, 200)
        max_depth = trial.suggest_int("max_depth", 2, 32)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
        
        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        # Evaluate using 5-Fold cross-validation
        scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy")
        return scores.mean()
        
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    return study.best_params`}
        </pre>
      </div>
    )
  },
  deployment: {
    title: 'Model Deployment & APIs',
    concept: (
      <div>
        <p><strong>What is Deployment?</strong></p>
        <p>Deploying a model means exposing the trained pipeline (including standard scaling, imputation, encoding, and the final model estimator) as an active REST API endpoint. This allows other programs or systems to query your model and retrieve predictions instantly.</p>
        
        <p style={{ marginTop: '1rem' }}><strong>Production Considerations:</strong></p>
        <ul style={{ paddingLeft: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
          <li><strong>Pipeline Serialization</strong>: The entire preprocessing and model pipeline is saved as a single file using `joblib` or `pickle`.</li>
          <li><strong>API Schema Validation</strong>: Client payloads must exactly match the feature schemas (data types, column names) of the training dataset.</li>
        </ul>
        {renderLinks([
          { text: "FastAPI Official Documentation", url: "https://fastapi.tiangolo.com/" },
          { text: "Deploying Machine Learning Models as APIs", url: "https://towardsdatascience.com/deploying-a-machine-learning-model-as-a-rest-api-4a03b4ad23c9" }
        ])}
      </div>
    ),
    math: (
      <div>
        <p><strong>API Prediction Pipeline:</strong></p>
        <p>The deployed model processes raw HTTP payloads using the saved pipeline transform:</p>
        <div style={{ padding: '0.75rem', backgroundColor: 'var(--bg-tertiary)', borderRadius: '6px', margin: '0.75rem 0', fontSize: '0.85rem', lineHeight: '1.6', fontFamily: 'var(--font-body)' }}>
          1. Receive JSON payload <LatexMath math="x_{raw}" />.<br />
          2. Impute missing features: <LatexMath math="x_{imputed} = \text{Imputer}(x_{raw})" />.<br />
          3. Scale numerical values: <LatexMath math="x_{scaled} = \text{Scaler}(x_{imputed})" />.<br />
          4. Predict target: <LatexMath math="y_{pred} = \text{Model.predict}(x_{scaled})" />.<br />
          5. Return prediction response to client.
        </div>
      </div>
    ),
    code: (
      <div>
        <p><strong>Backend Python Implementation (API Server):</strong></p>
        <pre style={{ backgroundColor: 'var(--bg-tertiary)', padding: '1rem', borderRadius: '6px', overflowX: 'auto', fontSize: '0.8rem', color: 'var(--text-main)', border: '1px solid var(--border-color)' }}>
{`import joblib
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()
# Load the pre-trained pipeline file
pipeline = joblib.load("model_pipeline.pkl")

class PredictionPayload(BaseModel):
    features: dict

@app.post("/predict")
def predict(payload: PredictionPayload):
    # Convert payload dict to pandas DataFrame
    df_input = pd.DataFrame([payload.features])
    
    # Execute full prediction pipeline
    prediction = pipeline.predict(df_input)
    return {"prediction": int(prediction[0])}`}
        </pre>
      </div>
    )
  }
};

const GuideDrawer = ({ isOpen, onClose, initialTopic }) => {
  const [activeTopic, setActiveTopic] = useState(initialTopic || 'data_upload');
  const [activeTab, setActiveTab] = useState('concept'); // concept, math, code

  useEffect(() => {
    if (initialTopic && guideTopics[initialTopic]) {
      setActiveTopic(initialTopic);
    }
  }, [initialTopic, isOpen]);

  const topicData = guideTopics[activeTopic] || guideTopics.data_upload;

  return (
    <>
      {/* Backdrop */}
      <div 
        className={`guide-drawer-backdrop ${isOpen ? 'open' : ''}`} 
        onClick={onClose}
        style={{
          position: 'fixed',
          top: 0,
          left: 0,
          width: '100vw',
          height: '100vh',
          backgroundColor: 'rgba(0, 0, 0, 0.45)',
          backdropFilter: 'blur(4px)',
          zIndex: 1000,
          opacity: isOpen ? 1 : 0,
          pointerEvents: isOpen ? 'auto' : 'none',
          transition: 'all 0.25s ease'
        }}
      />

      {/* Panel */}
      <div 
        className={`guide-drawer-panel ${isOpen ? 'open' : ''}`}
        style={{
          position: 'fixed',
          top: 0,
          right: 0,
          width: '750px',
          maxWidth: '92vw',
          height: '100vh',
          backgroundColor: 'var(--bg-secondary)',
          borderLeft: '1px solid var(--border-color)',
          boxShadow: '-10px 0 30px rgba(0, 0, 0, 0.5)',
          zIndex: 1001,
          transform: isOpen ? 'translateX(0)' : 'translateX(100%)',
          transition: 'transform 0.3s cubic-bezier(0.16, 1, 0.3, 1)',
          display: 'flex',
          flexDirection: 'column'
        }}
      >
        {/* Header */}
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1.25rem 1.5rem', borderBottom: '1px solid var(--border-color)', backgroundColor: 'var(--bg-tertiary)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <i className="fa-solid fa-graduation-cap" style={{ color: 'var(--accent-primary)', fontSize: '1.2rem' }}></i>
            <h2 style={{ fontSize: '1.1rem', fontWeight: 700, margin: 0, letterSpacing: '-0.02em', color: 'var(--text-main)' }}>InstaML Educational Guide</h2>
          </div>
          <button 
            onClick={onClose} 
            style={{ background: 'none', border: 'none', color: 'var(--text-muted)', fontSize: '1.2rem', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', padding: '0.25rem', borderRadius: '4px' }}
            onMouseEnter={(e) => e.currentTarget.style.color = 'var(--text-main)'}
            onMouseLeave={(e) => e.currentTarget.style.color = 'var(--text-muted)'}
          >
            <i className="fa-solid fa-xmark"></i>
          </button>
        </div>

        {/* Body Split */}
        <div style={{ display: 'flex', flex: 1, overflow: 'hidden' }}>
          {/* Inner Sidebar (Topics) */}
          <div style={{ width: '220px', borderRight: '1px solid var(--border-color)', backgroundColor: 'rgba(255, 255, 255, 0.005)', overflowY: 'auto', padding: '1rem 0.5rem' }}>
            <span style={{ fontSize: '0.65rem', fontWeight: 700, color: 'var(--text-dim)', textTransform: 'uppercase', letterSpacing: '0.05em', padding: '0 0.75rem', marginBottom: '0.5rem', display: 'block' }}>
              Select Topic
            </span>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.15rem' }}>
              {Object.keys(guideTopics).map(key => (
                <button
                  key={key}
                  onClick={() => { setActiveTopic(key); setActiveTab('concept'); }}
                  style={{
                    textAlign: 'left',
                    padding: '0.55rem 0.75rem',
                    borderRadius: '6px',
                    border: 'none',
                    fontSize: '0.8rem',
                    cursor: 'pointer',
                    fontWeight: activeTopic === key ? 600 : 500,
                    color: activeTopic === key ? 'var(--text-main)' : 'var(--text-muted)',
                    backgroundColor: activeTopic === key ? 'var(--bg-active)' : 'transparent',
                    transition: 'all 0.15s ease'
                  }}
                  onMouseEnter={(e) => {
                    if (activeTopic !== key) e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.015)';
                  }}
                  onMouseLeave={(e) => {
                    if (activeTopic !== key) e.currentTarget.style.backgroundColor = 'transparent';
                  }}
                >
                  {guideTopics[key].title}
                </button>
              ))}
            </div>
          </div>

          {/* Main Content Area */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column', overflow: 'hidden', backgroundColor: 'var(--bg-secondary)' }}>
            {/* Topic Title */}
            <div style={{ padding: '1.25rem 1.5rem 0 1.5rem' }}>
              <h3 style={{ fontSize: '1.2rem', fontWeight: 700, margin: '0 0 1rem 0', color: 'var(--text-main)', letterSpacing: '-0.025em' }}>
                {topicData.title}
              </h3>

              {/* Sub-Tabs */}
              <div style={{ display: 'flex', gap: '0.5rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.5rem' }}>
                <button 
                  className={`tab-btn tab-btn-sm ${activeTab === 'concept' ? 'active' : ''}`}
                  onClick={() => setActiveTab('concept')}
                  style={{
                    padding: '0.4rem 0.8rem',
                    fontSize: '0.75rem',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: 600,
                    backgroundColor: activeTab === 'concept' ? 'var(--bg-active)' : 'transparent',
                    color: activeTab === 'concept' ? 'var(--text-main)' : 'var(--text-muted)'
                  }}
                >
                  <i className="fa-solid fa-book-open" style={{ marginRight: '0.35rem' }}></i>
                  Concept & Rules
                </button>
                <button 
                  className={`tab-btn tab-btn-sm ${activeTab === 'math' ? 'active' : ''}`}
                  onClick={() => setActiveTab('math')}
                  style={{
                    padding: '0.4rem 0.8rem',
                    fontSize: '0.75rem',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: 600,
                    backgroundColor: activeTab === 'math' ? 'var(--bg-active)' : 'transparent',
                    color: activeTab === 'math' ? 'var(--text-main)' : 'var(--text-muted)'
                  }}
                >
                  <i className="fa-solid fa-square-root-variable" style={{ marginRight: '0.35rem' }}></i>
                  Mathematics
                </button>
                <button 
                  className={`tab-btn tab-btn-sm ${activeTab === 'code' ? 'active' : ''}`}
                  onClick={() => setActiveTab('code')}
                  style={{
                    padding: '0.4rem 0.8rem',
                    fontSize: '0.75rem',
                    border: 'none',
                    borderRadius: '4px',
                    cursor: 'pointer',
                    fontWeight: 600,
                    backgroundColor: activeTab === 'code' ? 'var(--bg-active)' : 'transparent',
                    color: activeTab === 'code' ? 'var(--text-main)' : 'var(--text-muted)'
                  }}
                >
                  <i className="fa-solid fa-code" style={{ marginRight: '0.35rem' }}></i>
                  Python Code
                </button>
              </div>
            </div>

            {/* Scrollable Content Container */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '1.5rem', lineHeight: '1.6', fontSize: '0.95rem', color: 'var(--text-main)' }}>
              {activeTab === 'concept' && topicData.concept}
              {activeTab === 'math' && topicData.math}
              {activeTab === 'code' && topicData.code}
            </div>
          </div>
        </div>
      </div>
    </>
  );
};

export default GuideDrawer;
