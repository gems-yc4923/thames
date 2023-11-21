import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score
from scipy.stats import randint as ran
from scipy.stats import uniform
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso

class ModelSelector:
    def __init__(self, data, target, complete_pipe, task='class', i=2):
        '''
        Parameters
        ----------
        data : pandas DataFrame
            The data to be used for model selection
        target : str
            The target column name
        complete_pipe : sklearn Pipeline
            The complete pipeline to be used for preprocessing
        task : str, optional
            The task to be performed, by default 'class', or 'reg'
        i : int, optional
            The number of best models to be selected, by default 2, Maximum 6. (The more the longer it takes to run)
        ----------
        Returns
        ----------
        Run get_pipeline() to get the final pipeline
        '''
        self.complete_pipe = complete_pipe
        self.i = i
        self.data = data
        self.data = self.data.sample(frac=0.2, random_state=42)
        self.task = task
        if self.task != 'class' and self.task != 'reg':
            raise ValueError("Task must be either 'class' or 'reg'")
        if self.i > 6:
            raise ValueError("i must be less than or equal to 6")
        self.target = target
        self.X = self.data.drop(self.target, axis=1)
        self.y = self.data[self.target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,random_state=42)
        self.le = LabelEncoder()
        self.final_model = None
        self.best_models = None

        # Define your models and parameters here
        self.models_parameters_classification = {
            'LogisticRegression': {
                'model': LogisticRegression(max_iter=3000),
                'params': {
                    'C': uniform(0.1, 10),
                    'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                    'class_weight': [None, 'balanced']
                }
            },
            'RandomForestClassifier' : {
            'model': RandomForestClassifier(),
                'params': {
                    'max_depth': ran(1,50),
                    'n_estimators': ran(100,500),
                    'min_samples_split': ran(2,10),
                    'max_features': ran(1,8),
                    'class_weight': [None, 'balanced', 'balanced_subsample']
                }
            },
            'KNeighborsClassifier': {
                'model': KNeighborsClassifier(),
                'params': {
                    'n_neighbors': ran(1,10),
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            },
            'DecisionTreeClassifier': {
                'model': DecisionTreeClassifier(),
                'params': {
                    'max_depth': ran(1,50),
                    'min_samples_split': ran(2,10),
                    'max_features': ran(1,8),
                    'class_weight': [None, 'balanced']
                }
            },
            'SVC': {
                'model': SVC(probability=True),
                'params': {
                    'C': uniform(0.1, 10),
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
                    'class_weight': [None, 'balanced']
                }
            },
                'MLPClassifier': {
                'model': MLPClassifier(random_state=42, max_iter=200),
                'params': {
                    'hidden_layer_sizes': [(50,), (100,), (50,50), (100,50)],
                    'activation': ['identity', 'logistic', 'tanh', 'relu'],
                    'solver': ['lbfgs', 'sgd', 'adam'],
                    'early_stopping': [True],
                    'alpha': uniform(0.0001, 0.001),
                    'learning_rate': ['constant', 'invscaling', 'adaptive'],
                }
            }
        }
        self.models_parameters_regression = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {
                    'fit_intercept': [True, False],
                    'normalize': [True, False]
                }
            },
            'RandomForestRegressor': {
                'model': RandomForestRegressor(),
                'params': {
                    'max_depth': ran(1, 50),
                    'n_estimators': ran(100, 500),
                    'min_samples_split': ran(2, 10),
                    'max_features': ran(1, 8)
                }
            },
            'SVR': {
                'model': SVR(),
                'params': {
                    'C': uniform(0.1, 10),
                    'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
                }
            },
            'KNeighborsRegressor': {
                'model': KNeighborsRegressor(),
                'params': {
                    'n_neighbors': ran(1, 10),
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
                }
            },
            'XGBRegressor': {
                'model': XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=3),
                'params': {
                    'n_estimators': ran(100, 500),
                    'max_depth': ran(3, 10),
                    'learning_rate': uniform(0.01, 0.3),
                    'subsample': uniform(0.5, 0.5)
                }
            },
            'LassoRegressor': {
                'model': Lasso(),
                'params': {
                    'alpha': uniform(0.01, 10),
                    'fit_intercept': [True, False],
                    'normalize': [True, False],
                    'max_iter': [1000, 2000, 3000],
                    'tol': [1e-4, 1e-3, 1e-2],
                    'selection': ['cyclic', 'random']
                }
            }
        }

    def fit(self):
        '''
        Fits the model selection pipeline
        '''
        # Preprocess and Label Encode y_train
        y_train_encoded = self.le.fit_transform(self.y_train)
        X_transformed = self.complete_pipe.fit_transform(self.X_train)
        # Model selection pipeline for classification
        if self.task == 'class':
            pipe = Pipeline([
                ('classifier', RandomForestClassifier())
            ])
            # GridSearchCV for model selection
            models = [{'classifier': [model_info['model']]} for _, model_info in self.models_parameters_classification.items()]
            self.result_model = GridSearchCV(pipe, models, cv=5, scoring='accuracy')
            self.result_model.fit(X_transformed, y_train_encoded)
            # Selecting best models
            self.select_best_classifier()
        # Model selection pipeline for regression
        elif self.task == 'reg':
            # Regression pipeline and GridSearchCV
            pipe = Pipeline([
                ('regressor', LinearRegression())
            ])
            models = [{'regressor': [model_info['model']]} for _, model_info in self.models_parameters_regression.items()]
            self.result_model = GridSearchCV(pipe, models, cv=5, scoring='r2')
            self.result_model.fit(self.complete_pipe.transform(self.X_train), self.y_train)
            self.select_best_regressor()

    def get_pipeline(self):
        '''
        Returns
        -------
        sklearn Pipeline
            The final pipeline containing preprocessing and the ensemble model
            '''
        return self.final_model  

    def select_best_classifier(self):
        # Extracting best models
        results = pd.DataFrame(self.result_model.cv_results_).sort_values(by='mean_test_score', ascending=False)
        self.best_models = (results['param_classifier'].iloc[0:self.i].values).tolist()

        # Fine-tuning best models
        final_params = []
        final_models = []
        for model in self.best_models:
            for model_name, model_info in self.models_parameters_classification.items():
                if isinstance(model, type(model_info['model'])):
                    params = model_info['params']
                    test_search = RandomizedSearchCV(
                        model,
                        param_distributions=params,
                        cv=5,
                        random_state=42,
                        n_iter=10,
                        n_jobs=-1,
                    )
                    test_search.fit(self.complete_pipe.transform(self.X_train), self.y_train)
                    final_params.append({test_search.best_estimator_: test_search.best_params_})
                    final_models.append((type(model).__name__,test_search.best_estimator_))
                    
                    # Printing the model name and its parameters
                    print(f"Model Type: {type(model).__name__}")
                    print()

        # Creating final ensemble model
        from sklearn.pipeline import make_pipeline
        self.ensemble = VotingClassifier(estimators= final_models, voting='soft')
        self.final_model = make_pipeline(self.complete_pipe, self.ensemble)

    def select_best_regressor(self):
        # Extracting best models
        results = pd.DataFrame(self.result_model.cv_results_).sort_values(by='mean_test_score', ascending=False)
        best_regressors = (results['param_regressor'].iloc[0:self.i].values).tolist()

        # Fine-tuning best models
        final_params = []
        final_models = []
        for model in best_regressors:
            for model_name, model_info in self.models_parameters_regression.items():
                if isinstance(model, type(model_info['model'])):
                    params = model_info['params']
                    test_search = RandomizedSearchCV(
                        model,
                        param_distributions=params,
                        cv=5,
                        random_state=42,
                        n_iter=10,
                        n_jobs=-1,
                    )
                    test_search.fit(self.complete_pipe.transform(self.X_train), self.y_train)
                    final_params.append({test_search.best_estimator_: test_search.best_params_})
                    final_models.append((type(model).__name__, test_search.best_estimator_))

                    # Printing the model name and its parameters
                    print(f"Model Type: {type(model).__name__}")
                    print()

        # Creating final ensemble model for regression
        from sklearn.pipeline import make_pipeline
        from sklearn.ensemble import VotingRegressor
        self.regressor_ensemble = VotingRegressor(estimators=final_models)
        self.final_regressor_model = make_pipeline(self.complete_pipe, self.regressor_ensemble)

    def evaluate(self):
        # Fit the final model and predict
        self.final_model.fit(self.X_train, self.y_train)
        y_pred = self.final_model.predict(self.X_test)

        # Evaluate the model
        final_score = accuracy_score(self.y_test, y_pred)
        final_precision = precision_score(self.y_test, y_pred)
        return final_score, final_precision

    def print_results(self):
        final_score, final_precision = self.evaluate()
        print(f"The Final score of our generated Model is: {final_score*100:.2f}%")
        print(f"The Precision of our generated Model is: {final_precision*100:.2f}%")