{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": []
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "execution_count": 3,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "MWl2x-MkqbLg",
        "outputId": "17d15ccd-3463-479a-b2e5-6bfeddf18699"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Boosting Accuracy: 0.875\n",
            "Bagging Accuracy: 0.885\n",
            "Random Forest Accuracy: 0.88\n"
          ]
        }
      ],
      "source": [
        "import numpy as np\n",
        "from sklearn.datasets import make_classification\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import AdaBoostClassifier , BaggingClassifier , RandomForestClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "x, y = make_classification(n_samples=1000 , n_features=20 , n_classes=2, random_state = 42)\n",
        "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)\n",
        "boosting_classfier = AdaBoostClassifier(n_estimators=50,random_state=42)\n",
        "BaggingClassifier = BaggingClassifier(n_estimators=50,random_state=42)\n",
        "random_forest_classifier = RandomForestClassifier(n_estimators=50,random_state=42)\n",
        "boosting_classfier.fit(x_train,y_train)\n",
        "BaggingClassifier.fit(x_train,y_train)\n",
        "random_forest_classifier.fit(x_train,y_train)\n",
        "boosting_pred = boosting_classfier.predict(x_test)\n",
        "bagging_pred = BaggingClassifier.predict(x_test)\n",
        "random_forest_pred = random_forest_classifier.predict(x_test)\n",
        "boosting_accuracy = accuracy_score(y_test,boosting_pred)\n",
        "bagging_accuracy = accuracy_score(y_test,bagging_pred)\n",
        "random_forest_accuracy = accuracy_score(y_test,random_forest_pred)\n",
        "print(\"Boosting Accuracy:\",boosting_accuracy)\n",
        "print(\"Bagging Accuracy:\",bagging_accuracy)\n",
        "print(\"Random Forest Accuracy:\",random_forest_accuracy)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Q2\n",
        "import numpy as np\n",
        "from sklearn.datasets import fetch_california_housing\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.ensemble import AdaBoostRegressor, BaggingRegressor, RandomForestRegressor\n",
        "from sklearn.metrics import r2_score, mean_squared_error\n",
        "# Fetch the California housing dataset\n",
        "housing = fetch_california_housing()\n",
        "x = housing.data\n",
        "y = housing.target\n",
        "# Split data into training and testing sets\n",
        "x_train, x_test, y_train, y_test = train_test_split(\n",
        " x, y, test_size=0.2, random_state=42\n",
        ")\n",
        "# Initialize regressors\n",
        "boosting_regressor = AdaBoostRegressor(n_estimators=50, random_state=42)\n",
        "bagging_regressor = BaggingRegressor(n_estimators=50, random_state=42)\n",
        "random_forest_regressor = RandomForestRegressor(n_estimators=50, random_state=42)\n",
        "# Fit the models\n",
        "boosting_regressor.fit(x_train, y_train)\n",
        "bagging_regressor.fit(x_train, y_train)\n",
        "random_forest_regressor.fit(x_train, y_train)\n",
        "# Make predictions (missing in original code)\n",
        "boosting_pred = boosting_regressor.predict(x_test)\n",
        "bagging_pred = bagging_regressor.predict(x_test)\n",
        "random_forest_pred = random_forest_regressor.predict(x_test)\n",
        "# Calculate R2 scores\n",
        "boosting_r2 = r2_score(y_test, boosting_pred)\n",
        "bagging_r2 = r2_score(y_test, bagging_pred)\n",
        "random_forest_r2 = r2_score(y_test, random_forest_pred)\n",
        "# Calculate Mean Squared Errors\n",
        "boosting_mse = mean_squared_error(y_test, boosting_pred)\n",
        "bagging_mse = mean_squared_error(y_test, bagging_pred)\n",
        "random_forest_mse = mean_squared_error(y_test, random_forest_pred)\n",
        "# Print the results\n",
        "print(\"Boosting R2 Score:\", boosting_r2)\n",
        "print(\"Bagging R2 Score:\", bagging_r2)\n",
        "print(\"Random Forest R2 Score:\", random_forest_r2)\n",
        "print(\"\\nAdaBoost Regressor MSE:\", boosting_mse)\n",
        "print(\"Bagging Regressor MSE:\", bagging_mse)\n",
        "print(\"Random Forest Regressor MSE:\", random_forest_mse)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ey-5KLOUrIAD",
        "outputId": "4ebe083c-192f-4021-f3ae-f0d259131018"
      },
      "execution_count": 5,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Boosting R2 Score: 0.5310788695468394\n",
            "Bagging R2 Score: 0.8036499747356253\n",
            "Random Forest R2 Score: 0.8036506665860602\n",
            "\n",
            "AdaBoost Regressor MSE: 0.614478459432694\n",
            "Bagging Regressor MSE: 0.2572988359842641\n",
            "Random Forest Regressor MSE: 0.2572979293772426\n"
          ]
        }
      ]
    }
  ]
}