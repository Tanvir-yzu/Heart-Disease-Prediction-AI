import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBClassifier
import joblib

# 1. ডেটা লোড
df = pd.read_csv("heart.csv")  # নিশ্চিত করুন যে ফাইলটি একই ফোল্ডারে আছে

# 2. টার্গেট এবং ফিচার
X = df.drop(columns=["HeartDisease"])  # টার্গেট কলামের নাম যদি ভিন্ন হয়, তবে ঠিক করুন
y = df["HeartDisease"]

# 3. Train-test split (এখানে সেভ করার জন্য প্রয়োজন নেই, কিন্তু ভবিষ্যতে ভালো)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. কলাম টাইপ নির্ধারণ
numeric_cols = ["Age", "RestingBP", "Cholesterol", "FastingBS", "MaxHR", "Oldpeak"]
categorical_cols = ["Sex", "ChestPainType", "RestingECG", "ExerciseAngina", "ST_Slope"]

# 5. Preprocessor Pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ])

# 6. Full Pipeline (প্রি-প্রসেসিং + মডেল)
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(random_state=42))
])

# 7. মডেল ট্রেইন
model.fit(X_train, y_train)

# 8. মডেল সেভ করা (.pkl)
joblib.dump(model, 'xgboost_heart_failure_model.pkl')

print("✅ Model trained and saved as 'xgboost_heart_failure_model.pkl'")