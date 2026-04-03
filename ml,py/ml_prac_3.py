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
      "execution_count": 4,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "pSAlE6NplJLz",
        "outputId": "2473f1d3-10a3-41be-9e06-5cebccd31ea1"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Using Colab cache for faster access to the 'smarket-data' dataset.\n",
            "Accuracy 0.492\n"
          ]
        }
      ],
      "source": [
        "#Q1#smarket dataset\n",
        "import os\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "import kagglehub\n",
        "path = kagglehub.dataset_download(\"mousmifarouq/smarket-data\")\n",
        "csv_file_path = os.path.join(path, \"Smarket data.csv\")\n",
        "smarket_df = pd.read_csv(csv_file_path)\n",
        "x=smarket_df[['Lag1','Lag2','Lag3','Lag4','Lag5','Volume']].values\n",
        "y=smarket_df['Direction'].map({'Up':1,'Down':0}).values\n",
        "x_train , x_test ,y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)\n",
        "scaler = StandardScaler()\n",
        "x_train_scaled = scaler.fit_transform(x_train)\n",
        "x_test_scaled = scaler.fit_transform(x_test)\n",
        "k=3\n",
        "knn_model = KNeighborsClassifier(n_neighbors=k)\n",
        "knn_model.fit(x_train_scaled , y_train)\n",
        "y_pred = knn_model.predict(x_test_scaled)\n",
        "accuracy = accuracy_score(y_test,y_pred)\n",
        "print(\"Accuracy\",accuracy)"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Q2\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "import kagglehub\n",
        "path = kagglehub.dataset_download(\"mousmifarouq/smarket-data\")\n",
        "x=smarket_df[['Year','Today']].values\n",
        "y=smarket_df['Direction'].map({'Up':1,'Down':0}).values\n",
        "x_train , x_test ,y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=42)\n",
        "scaler = StandardScaler()\n",
        "x_train_scaled = scaler.fit_transform(x_train)\n",
        "x_test_scaled = scaler.transform(x_test)\n",
        "k=5\n",
        "knn_model = KNeighborsClassifier(n_neighbors=k)\n",
        "knn_model.fit(x_train_scaled , y_train)\n",
        "y_pred = knn_model.predict(x_test_scaled)\n",
        "accuracy = accuracy_score(y_test,y_pred)\n",
        "print(\"Accuracy\",accuracy)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bC1r5SH4mXXV",
        "outputId": "aca772b1-f427-4ab5-e6af-91a59d745544"
      },
      "execution_count": 9,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Using Colab cache for faster access to the 'smarket-data' dataset.\n",
            "Accuracy 1.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Q3 iris dataset\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.datasets import load_iris\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "iris = load_iris()\n",
        "x = iris.data\n",
        "y = iris.target\n",
        "x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)\n",
        "knn = KNeighborsClassifier(n_neighbors=3)\n",
        "knn.fit(x_train,y_train)\n",
        "y_pred = knn.predict(x_test)\n",
        "accuracy = accuracy_score(y_test,y_pred)\n",
        "print(\"Accuracy: \", accuracy)\n"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "bHXEsgUMoIcV",
        "outputId": "a0d5b2dc-88b0-4f37-8950-8d00e70cee99"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy:  1.0\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "# Q4\n",
        "# breast cancer data set\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.datasets import load_breast_cancer\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "cancer=load_breast_cancer()\n",
        "x=cancer.data\n",
        "y=cancer.target\n",
        "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)\n",
        "knn=KNeighborsClassifier(n_neighbors=5)\n",
        "knn.fit(x_train,y_train)\n",
        "y_pred=knn.predict(x_test)\n",
        "accuracy=accuracy_score(y_test,y_pred)\n",
        "print(\"Accuracy:\",accuracy)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "-mjl7643oxQE",
        "outputId": "64db59b8-92d0-4e6d-8d07-a3b2458908a3"
      },
      "execution_count": 10,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.956140350877193\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "#Q5\n",
        "#glass dataset\n",
        "import pandas as pd\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.metrics import accuracy_score\n",
        "\n",
        "# Corrected: Load the CSV without a header and assign column names\n",
        "# The glass dataset typically has 10 columns, with the last one being the type.\n",
        "column_names = ['RI', 'Na', 'Mg', 'Al', 'Si', 'K', 'Ca', 'Ba', 'Fe', 'Type']\n",
        "glass_data = pd.read_csv(\"/glass.csv\", header=None)\n",
        "glass_data.columns = column_names\n",
        "\n",
        "x = glass_data.drop('Type', axis=1)\n",
        "y = glass_data['Type']\n",
        "x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)\n",
        "knn=KNeighborsClassifier(n_neighbors=5)\n",
        "knn.fit(x_train,y_train)\n",
        "y_pred = knn.predict(x_test)\n",
        "accuracy=accuracy_score(y_test,y_pred)\n",
        "print(\"Accuracy:\",accuracy)"
      ],
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "DKKuQhESo6-L",
        "outputId": "cd2bf507-71ec-4c1a-e1b1-095675bec348"
      },
      "execution_count": 14,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Accuracy: 0.6511627906976745\n"
          ]
        }
      ]
    }
  ]
}