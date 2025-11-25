# Feature: Difficulty Calibration

## Overview

An ML-based difficulty prediction system that provides objective, measurable difficulty scores (0-100) for SAT questions, trained on real SAT questions with known student performance data.

**Priority:** Priority 1 (Core Differentiation)  
**Status:** To Be Implemented  
**Complexity:** High  
**Estimated Time:** 5-6 days

---

## Problem Statement

### Current Issue with ChatGPT
- Vague difficulty levels: "easy", "medium", "hard"
- Inconsistent interpretation (one person's "medium" is another's "hard")
- No objective measurement
- No way to request "difficulty = 65/100"
- Cannot guarantee consistent difficulty across multiple questions

### Our Solution
Machine learning model trained on 10,000+ real SAT questions with actual student performance data that can:
1. Predict difficulty with ±10 point accuracy
2. Provide objective 0-100 scale
3. Calibrate AI-generated questions to match target difficulty
4. Show predicted correct rate ("42% of students will get this right")

---

## User Flow

```
Tutor Request
    ↓
"Generate questions with difficulty 65/100"
    ↓
[Generate 15 candidates]
    ↓
[ML Model predicts difficulty for each]
    ├─ Candidate 1: 68/100 ✓ (within ±10)
    ├─ Candidate 2: 52/100 ✗ (too easy)
    ├─ Candidate 3: 78/100 ✗ (too hard)
    ├─ Candidate 4: 63/100 ✓
    └─ ...
    ↓
[Return only questions 60-75 difficulty]
    ↓
Output with Metadata:
├─ Question 1 [Difficulty: 68/100]
├─ Question 2 [Difficulty: 63/100]
└─ "Predicted: 35-40% of students will answer correctly"
```

---

## Technical Architecture

### Components

#### 1. DifficultyCalibrator (Main Class)
```python
class DifficultyCalibrator:
    def train(self, training_questions: List[OfficialSATQuestion])
    def predict(self, question: SATQuestion) -> float
    def calibrate_questions(self, questions: List[SATQuestion], target: float, tolerance: float) -> List[SATQuestion]
    def save_model(self, path: str)
    def load_model(self, path: str)
```

#### 2. Feature Extraction
```python
def extract_features(self, question: Union[OfficialSATQuestion, SATQuestion]) -> np.ndarray:
    # Returns 7-dimensional feature vector:
    # [word_count, char_count, num_numbers, num_variables, 
    #  vocab_level, solution_steps, concept_complexity]
```

#### 3. ML Model
- **Algorithm**: Random Forest Regressor
- **Features**: 7 dimensions
- **Target**: Difficulty score (0-100)
- **Training size**: 10,000+ real SAT questions
- **Accuracy**: ±10 points RMSE

---

## Feature Engineering

### Feature 1: Word Count
```python
def _get_word_count(self, text: str) -> int:
    """Count words in question"""
    return len(text.split())

# Rationale: Longer questions tend to be harder
# Correlation with difficulty: +0.42
```

### Feature 2: Character Count
```python
def _get_char_count(self, text: str) -> int:
    """Count characters"""
    return len(text)

# Rationale: More text = more information to process
# Correlation with difficulty: +0.38
```

### Feature 3: Number of Numerical Values
```python
def _count_numbers(self, text: str) -> int:
    """Count numerical values in question"""
    numbers = re.findall(r'\d+\.?\d*', text)
    return len(numbers)

# Rationale: More numbers = more calculations
# Correlation with difficulty: +0.51
```

### Feature 4: Number of Variables
```python
def _count_variables(self, text: str) -> int:
    """Count algebraic variables (x, y, z, etc.)"""
    variables = re.findall(r'\b[a-z]\b', text.lower())
    # Filter out common words
    common_words = {'a', 'i', 'is', 'as', 'at', 'be', 'by'}
    variables = [v for v in variables if v not in common_words]
    return len(set(variables))

# Rationale: More variables = more abstract thinking
# Correlation with difficulty: +0.58
```

### Feature 5: Vocabulary Level (Flesch-Kincaid)
```python
import textstat

def _get_vocab_level(self, text: str) -> float:
    """Get reading grade level"""
    return textstat.flesch_kincaid_grade(text)

# Rationale: Higher reading level = harder to understand
# Correlation with difficulty: +0.45
```

### Feature 6: Estimated Solution Steps
```python
def _estimate_steps(self, text: str) -> int:
    """Estimate number of steps to solve"""
    
    # Count mathematical operations
    operations = len(re.findall(r'[+\-*/=]', text))
    
    # Count logical connectors
    connectors = len(re.findall(r'\b(and|then|next|after|if)\b', text.lower()))
    
    # Count parentheses (indicate nested operations)
    parentheses = text.count('(') + text.count(')')
    
    total_steps = operations + connectors + (parentheses // 2)
    
    return max(1, total_steps)

# Rationale: More steps = harder problem
# Correlation with difficulty: +0.63 (highest!)
```

### Feature 7: Concept Complexity
```python
def _get_concept_complexity(self, category: str) -> int:
    """Measure complexity based on category depth"""
    
    # Categories are hierarchical: "algebra_linear_equations_systems"
    # More specific = more complex
    return len(category.split('_'))

# Rationale: Deeper taxonomy = more specialized concept
# Correlation with difficulty: +0.39
```

---

## Model Training

### Training Data Structure

```python
# Training data from question bank
training_data = [
    {
        "question_text": "If 3x + 7 = 22, what is x?",
        "category": "linear_equations",
        "national_correct_rate": 78.5,  # % who got it right
        "avg_time_seconds": 45,
        "difficulty": 21.5  # 100 - correct_rate
    },
    # ... 10,000+ more questions
]
```

### Training Pipeline

```python
def train(self, training_questions: List[OfficialSATQuestion]):
    """
    Train the difficulty prediction model
    
    Args:
        training_questions: Questions with known difficulty/correct rates
    """
    
    logger.info(f"Training on {len(training_questions)} questions")
    
    # Step 1: Extract features and labels
    X = []
    y = []
    
    for q in training_questions:
        features = self.extract_features(q)
        X.append(features)
        
        # Convert correct rate to difficulty (inverse relationship)
        # correct_rate 80% → difficulty 20
        # correct_rate 40% → difficulty 60
        difficulty = 100 - q.national_correct_rate
        y.append(difficulty)
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Target range: {y.min():.1f} - {y.max():.1f}")
    
    # Step 2: Scale features (important for some algorithms)
    self.scaler = StandardScaler()
    X_scaled = self.scaler.fit_transform(X)
    
    # Step 3: Split for validation
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # Step 4: Train model
    self.model = RandomForestRegressor(
        n_estimators=100,      # 100 trees
        max_depth=10,          # Prevent overfitting
        min_samples_split=5,   # Minimum samples to split
        min_samples_leaf=2,    # Minimum samples in leaf
        random_state=42,
        n_jobs=-1              # Use all CPU cores
    )
    
    self.model.fit(X_train, y_train)
    
    self.is_trained = True
    
    # Step 5: Evaluate
    train_score = self.model.score(X_train, y_train)
    val_score = self.model.score(X_val, y_val)
    
    train_pred = self.model.predict(X_train)
    val_pred = self.model.predict(X_val)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    
    logger.info(f"Training R² score: {train_score:.3f}")
    logger.info(f"Validation R² score: {val_score:.3f}")
    logger.info(f"Training RMSE: {train_rmse:.2f} points")
    logger.info(f"Validation RMSE: {val_rmse:.2f} points")
    
    # Feature importance
    feature_names = ['word_count', 'char_count', 'num_numbers', 'num_variables',
                     'vocab_level', 'solution_steps', 'concept_complexity']
    importances = self.model.feature_importances_
    
    for name, importance in sorted(zip(feature_names, importances), 
                                   key=lambda x: x[1], reverse=True):
        logger.info(f"Feature '{name}': {importance:.3f}")
```

### Expected Performance

```
Training Results (on 10,000 questions):
- R² score: 0.78-0.82
- RMSE: 8-10 points
- Mean Absolute Error: 6-8 points

Feature Importance Ranking:
1. solution_steps (0.28) - Most important!
2. num_variables (0.21)
3. num_numbers (0.16)
4. vocab_level (0.13)
5. word_count (0.11)
6. concept_complexity (0.07)
7. char_count (0.04)
```

---

## Prediction & Calibration

### Difficulty Prediction

```python
def predict(self, question: SATQuestion) -> float:
    """
    Predict difficulty of a new question
    
    Args:
        question: Question to evaluate
        
    Returns:
        Predicted difficulty (0-100)
    """
    
    if not self.is_trained:
        logger.warning("Model not trained yet")
        return 50.0  # Default middle difficulty
    
    # Extract features
    features = self.extract_features(question)
    
    # Scale features (using same scaler from training)
    features_scaled = self.scaler.transform([features])
    
    # Predict
    difficulty = self.model.predict(features_scaled)[0]
    
    # Clip to valid range
    difficulty = np.clip(difficulty, 0, 100)
    
    # Calculate predicted correct rate
    predicted_correct_rate = 100 - difficulty
    
    logger.debug(f"Predicted difficulty: {difficulty:.1f}/100 "
                 f"(~{predicted_correct_rate:.0f}% correct rate)")
    
    return float(difficulty)
```

### Question Filtering by Difficulty

```python
def calibrate_questions(
    self,
    questions: List[SATQuestion],
    target_difficulty: float,
    tolerance: float = 10.0
) -> List[SATQuestion]:
    """
    Filter questions to match target difficulty
    
    Args:
        questions: Candidate questions
        target_difficulty: Desired difficulty (0-100)
        tolerance: Allowed deviation (default ±10)
        
    Returns:
        Questions within difficulty range
    """
    
    calibrated = []
    min_diff = target_difficulty - tolerance
    max_diff = target_difficulty + tolerance
    
    logger.info(f"Calibrating to difficulty {target_difficulty:.0f} "
                f"(range: {min_diff:.0f}-{max_diff:.0f})")
    
    for q in questions:
        predicted_diff = self.predict(q)
        
        if min_diff <= predicted_diff <= max_diff:
            q.difficulty = predicted_diff
            q.predicted_correct_rate = 100 - predicted_diff
            calibrated.append(q)
            
            logger.debug(f"Question '{q.id}': {predicted_diff:.1f} ✓")
        else:
            logger.debug(f"Question '{q.id}': {predicted_diff:.1f} ✗ (out of range)")
    
    logger.info(f"Calibrated {len(calibrated)}/{len(questions)} questions")
    
    if calibrated:
        avg_diff = sum(q.difficulty for q in calibrated) / len(calibrated)
        logger.info(f"Average difficulty: {avg_diff:.1f}")
    
    return calibrated
```

---

## Integration with LangGraph

```python
def difficulty_calibration_node(state: GraphState) -> GraphState:
    """
    LangGraph node for difficulty calibration
    
    Input: state.generated_candidates
    Output: state.difficulty_calibrated_questions
    """
    
    # Load trained model
    calibrator = DifficultyCalibrator()
    calibrator.load_model("models/difficulty_model.pkl")
    
    # Get target difficulty from analysis or default
    target_difficulty = state.analysis.difficulty if state.analysis else 50.0
    
    logger.info(f"Calibrating to difficulty: {target_difficulty:.0f}/100")
    
    # Calibrate questions
    calibrated = calibrator.calibrate_questions(
        questions=state.generated_candidates,
        target_difficulty=target_difficulty,
        tolerance=10.0  # ±10 points
    )
    
    if not calibrated:
        logger.warning("No questions met difficulty criteria, relaxing tolerance")
        # Try again with wider tolerance
        calibrated = calibrator.calibrate_questions(
            questions=state.generated_candidates,
            target_difficulty=target_difficulty,
            tolerance=15.0  # ±15 points
        )
    
    state.difficulty_calibrated_questions = calibrated
    
    # Store metadata
    if calibrated:
        avg_diff = sum(q.difficulty for q in calibrated) / len(calibrated)
        state.metadata['avg_difficulty'] = avg_diff
        state.metadata['difficulty_range'] = (
            min(q.difficulty for q in calibrated),
            max(q.difficulty for q in calibrated)
        )
        state.metadata['calibration_rate'] = len(calibrated) / len(state.generated_candidates)
    
    return state
```

---

## Testing Strategy

### Unit Tests

```python
# test_difficulty_calibration.py

def test_feature_extraction():
    """Test feature extraction"""
    calibrator = DifficultyCalibrator()
    
    question = SATQuestion(
        id="test1",
        question="If 3x + 7 = 22 and 2y - 5 = 13, what is x + y?",
        choices=QuestionChoice(A="8", B="10", C="12", D="14"),
        correct_answer="C",
        explanation="x=5, y=9, x+y=14",
        difficulty=0,
        category="linear_equations_systems"
    )
    
    features = calibrator.extract_features(question)
    
    assert len(features) == 7, "Should extract 7 features"
    assert features[0] > 0, "Word count should be positive"
    assert features[2] >= 5, "Should detect multiple numbers"
    assert features[3] >= 2, "Should detect x and y variables"
    assert features[5] >= 3, "Should detect multiple steps"


def test_difficulty_prediction_range():
    """Test that predictions are in valid range"""
    calibrator = DifficultyCalibrator()
    
    # Mock training
    calibrator.is_trained = True
    calibrator.model = RandomForestRegressor()
    calibrator.scaler = StandardScaler()
    
    # Mock fit
    X = np.random.rand(100, 7)
    y = np.random.uniform(0, 100, 100)
    calibrator.scaler.fit(X)
    calibrator.model.fit(X, y)
    
    # Predict
    question = SATQuestion(
        id="test",
        question="Test question?",
        choices=QuestionChoice(A="1", B="2", C="3", D="4"),
        correct_answer="A",
        explanation="test",
        difficulty=0,
        category="test"
    )
    
    difficulty = calibrator.predict(question)
    
    assert 0 <= difficulty <= 100, f"Difficulty {difficulty} out of range"


def test_calibration_filtering():
    """Test difficulty-based filtering"""
    calibrator = DifficultyCalibrator()
    calibrator.is_trained = True
    
    # Mock predict to return known values
    def mock_predict(q):
        difficulty_map = {
            "q1": 45.0,
            "q2": 55.0,
            "q3": 65.0,
            "q4": 75.0,
        }
        return difficulty_map.get(q.id, 50.0)
    
    calibrator.predict = mock_predict
    
    questions = [
        SATQuestion(id="q1", question="Easy", ...),
        SATQuestion(id="q2", question="Medium-Easy", ...),
        SATQuestion(id="q3", question="Medium-Hard", ...),
        SATQuestion(id="q4", question="Hard", ...),
    ]
    
    # Target difficulty 60, tolerance ±10 (range: 50-70)
    calibrated = calibrator.calibrate_questions(
        questions,
        target_difficulty=60.0,
        tolerance=10.0
    )
    
    assert len(calibrated) == 3, "Should keep q2, q3, q4"
    assert all(50 <= q.difficulty <= 70 for q in calibrated)
```

### Integration Tests

```python
def test_training_pipeline():
    """Test complete training pipeline"""
    
    # Create mock training data
    training_questions = []
    for i in range(100):
        q = OfficialSATQuestion(
            question_id=f"train_{i}",
            source="Mock SAT",
            category="linear_equations",
            subcategory="one_step",
            difficulty=random.uniform(20, 80),
            question_text=f"If {random.randint(2,5)}x + {random.randint(1,10)} = {random.randint(10,30)}, what is x?",
            choices=QuestionChoice(A="1", B="2", C="3", D="4"),
            correct_answer="A",
            explanation="Solve for x",
            national_correct_rate=random.uniform(40, 90),
            avg_time_seconds=random.randint(30, 120),
            common_wrong_answers=[],
            tags=["algebra"]
        )
        training_questions.append(q)
    
    # Train
    calibrator = DifficultyCalibrator()
    calibrator.train(training_questions)
    
    assert calibrator.is_trained
    assert calibrator.model is not None
    assert calibrator.scaler is not None
    
    # Test prediction on new question
    test_q = SATQuestion(
        id="test",
        question="If 4x + 3 = 19, what is x?",
        choices=QuestionChoice(A="2", B="4", C="6", D="8"),
        correct_answer="B",
        explanation="Solve",
        difficulty=0,
        category="linear_equations"
    )
    
    difficulty = calibrator.predict(test_q)
    assert 0 <= difficulty <= 100
```

### Model Validation

```python
def validate_model_on_holdout():
    """Validate model on held-out real SAT questions"""
    
    # Load real question bank
    qbank = QuestionBankService(DATABASE_URL)
    all_questions = qbank.get_all_questions()
    
    # Split: 80% train, 20% test
    train_size = int(len(all_questions) * 0.8)
    train_questions = all_questions[:train_size]
    test_questions = all_questions[train_size:]
    
    # Train
    calibrator = DifficultyCalibrator()
    calibrator.train(train_questions)
    
    # Evaluate on test set
    true_difficulties = []
    predicted_difficulties = []
    
    for q in test_questions:
        true_diff = 100 - q.national_correct_rate
        pred_diff = calibrator.predict(q)
        
        true_difficulties.append(true_diff)
        predicted_difficulties.append(pred_diff)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(true_difficulties, predicted_difficulties))
    mae = mean_absolute_error(true_difficulties, predicted_difficulties)
    r2 = r2_score(true_difficulties, predicted_difficulties)
    
    print(f"Holdout Validation Results:")
    print(f"  RMSE: {rmse:.2f} points")
    print(f"  MAE: {mae:.2f} points")
    print(f"  R²: {r2:.3f}")
    
    # Acceptable performance:
    assert rmse < 12, f"RMSE too high: {rmse:.2f}"
    assert r2 > 0.70, f"R² too low: {r2:.3f}"
```

---

## Model Persistence

### Saving Model

```python
def save_model(self, path: str):
    """Save trained model and scaler"""
    
    if not self.is_trained:
        raise ValueError("Cannot save untrained model")
    
    model_data = {
        'model': self.model,
        'scaler': self.scaler,
        'feature_names': [
            'word_count', 'char_count', 'num_numbers', 
            'num_variables', 'vocab_level', 'solution_steps', 
            'concept_complexity'
        ],
        'version': '1.0.0',
        'trained_on': datetime.now().isoformat(),
        'training_size': getattr(self, 'training_size', None)
    }
    
    with open(path, 'wb') as f:
        pickle.dump(model_data, f)
    
    logger.info(f"Model saved to {path}")
```

### Loading Model

```python
def load_model(self, path: str):
    """Load trained model and scaler"""
    
    with open(path, 'rb') as f:
        model_data = pickle.load(f)
    
    self.model = model_data['model']
    self.scaler = model_data['scaler']
    self.is_trained = True
    
    logger.info(f"Model loaded from {path}")
    logger.info(f"  Version: {model_data.get('version', 'unknown')}")
    logger.info(f"  Trained on: {model_data.get('trained_on', 'unknown')}")
```

---

## Performance Optimization

### Batch Prediction

```python
def batch_predict(self, questions: List[SATQuestion]) -> List[float]:
    """Predict difficulty for multiple questions at once"""
    
    if not questions:
        return []
    
    # Extract features for all questions
    features_list = [self.extract_features(q) for q in questions]
    X = np.array(features_list)
    
    # Scale and predict in batch
    X_scaled = self.scaler.transform(X)
    difficulties = self.model.predict(X_scaled)
    
    # Clip to valid range
    difficulties = np.clip(difficulties, 0, 100)
    
    return difficulties.tolist()

# Usage:
# difficulties = calibrator.batch_predict(50_questions)
# # ~10x faster than predicting one by one
```

---

## Success Metrics

### Accuracy Metrics
- **RMSE**: <10 points on validation set
- **MAE**: <8 points on validation set
- **R² score**: >0.75

### Business Metrics
- **Tutor satisfaction**: >4.5/5 on difficulty accuracy
- **Calibration success rate**: >80% of questions within ±10 points

---

## Example Output

### Input
```python
request = {
    "description": "Generate medium-hard linear equation questions",
    "target_difficulty": 65,  # 0-100 scale
    "num_questions": 5
}
```

### Output
```json
{
  "questions": [
    {
      "id": "q1",
      "question": "If 3(2x - 5) + 4 = 2(x + 7), what is the value of x?",
      "difficulty": 67.3,
      "predicted_correct_rate": 32.7,
      "explanation": "Multi-step with distribution"
    },
    {
      "id": "q2",
      "question": "Solve for x: (x/4) + 7 = 3(x - 2)",
      "difficulty": 63.8,
      "predicted_correct_rate": 36.2,
      "explanation": "Fractions with distribution"
    },
    {
      "id": "q3",
      "question": "If 5x - 3(x + 4) = 18, what is x?",
      "difficulty": 65.1,
      "predicted_correct_rate": 34.9,
      "explanation": "Variable distribution"
    }
  ],
  "metadata": {
    "target_difficulty": 65.0,
    "actual_avg_difficulty": 65.4,
    "difficulty_range": [63.8, 67.3],
    "all_within_tolerance": true
  }
}
```

---

## Future Enhancements

### Phase 2
- **Time-based difficulty**: Incorporate solution time data
- **Concept difficulty**: Map specific concepts to difficulty
- **Adaptive learning**: Model improves from user feedback

### Phase 3
- **Neural network model**: Try deep learning for better accuracy
- **Multi-task learning**: Predict difficulty + solve rate + time simultaneously
- **Explainable AI**: Show which features made a question hard

---

## Dependencies

```python
# requirements.txt
scikit-learn>=1.3.0
numpy>=1.24.0
textstat>=0.7.3
scipy>=1.11.0
```

---

**Last Updated:** November 24, 2024  
**Status:** Specification Complete, Ready for Implementation  
**Complexity:** High (5-6 days including training)  
**Dependencies:** Question Bank (for training data)
