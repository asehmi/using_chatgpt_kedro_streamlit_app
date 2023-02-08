# Using ChatGPT to build a Kedro ML pipeline and Streamlit frontend

![App Screen Shot](https://raw.githubusercontent.com/asehmi/using_chatgpt_kedro_streamlit_app/main/images/screenshots.png)

    date: "2023-02-07"
    author:
        name: "Arvindra Sehmi"
        url: "https://www.linkedin.com/in/asehmi/"
        mail: "vin [at] thesehmis.com"
        avatar: "https://twitter.com/asehmi/profile_image?size=original"
    related:
        N/A

### Introduction

I recently came across an open-source Python DevOps framework [Kedro](https://kedro.org/) and thought, “Why not  have [ChatGPT](https://chat.openai.com/chat) teach me how to use it to build some ML/DevOps automation?” The idea was to:
1. Ask ChatGPT some basic questions about Kedro.
2. Ask it to use more advanced features in the Kedro framework.
3. Write my questions with hints and phrases that encouraged explanations of advanced Kedro features (to evolve incrementally as if I were taught by a teacher).

Kedro has some pipeline visualization capabilities, so I wondered:
- Could ChatGPT show me how to display pipeline graphs in Streamlit?
- Could ChatGPT build me an example ML model and explicitly refer to it in the Kedro pipeline?
- What does it take to scale the pipeline, and perform pipeline logging, monitoring, and error handling?
- Could I connect Kedro logs to a cloud-based logging service?
- Could ChatGPT contrast Kedro with similar (competing) products and services and show me how the pipeline it developed earlier could be implemented in one of them?

I wrote a blog post with annotated responses to the answers I got to my questions. I was super impressed and decided to implement the Kedro pipeline and Streamlit application as planned from what I learned. This repository contains all the code for the application. 

### Try the Streamlit app yourself

The application can be seen running in the Streamlit Cloud at the link below:

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://asehmi-using-chatgpt-kedro-streamlit-ap-srcstreamlit-app-14bz3i.streamlit.app/)

- The source OCLH crypto currency data is supplied in a single CSV file, and was previously downloaded from the Bitfinex exchange
- OCLH data is for 4 coins spanning the period June 1, 2022 to December 31, 2022
- OCLH data is in 15min frequency
- A Kedro data catalog of source and feature datasets is built for each coin and subsequently used in the Kedro ML pipeline
- You can run the Kedro ML pipeline to train, test and evaluate a Linear Regression model to predict next period (t+1) close prices from several feature techical indicators derived from the close price and volume 
- You can visualize candlestick and line charts for the source and feature datasets, by coin
- Run locally, you can visualize an interactive graph representation of the Kedro pipeline in the Streamlit application
- You can run the pipeline nodes and the pipeline visualization from the command line too, using Kedro's CLI tools

For Streamlit beginners, this aplication can be useful to learn how to:
- Structure a multipage application
- Use session state
- Use widget callbacks
- Use many different widgets
- Launch sub-processes
- Embed external GUIs
- Cache data and clear caches
- Plotly charting
- (Check out my [gists](https://gist.github.com/asehmi) for more Streamlit goodies)

## Installation

(_On Windows replace forward slashes with back slashes._)

Clone this repository, then install package requirements:

```bash
$ cd using_chatgpt_kedro_streamlit_app
$ pip install -r src/requirements.txt
```

## Usage

**Run the Streamlit app**:

```bash
$ cd using_chatgpt_kedro_streamlit_app
$ streamlit run --server.port=2023 src/streamlit_app.py
```

**Run the Kedo pipeline from the command line**:

```bash
$ cd using_chatgpt_kedro_streamlit_app
$ kedro run
```

You should see a trace similar to this:

<details>
  <summary>Kedro run output trace</summary>

    🥁 Running from Kedro's CLI
    #### Pipeline execution order ####
    Inputs: uni_crypto_features_data

    Get-Current-Symbol
    Train-and-Test-Data-Split
    Model-Training
    Model-Evaluation
    Display-Model-Evaluation-Metrics

    Outputs: None
    ##################################
    [02/07/23 13:28:06] INFO     Loading data from 'uni_crypto_features_data' (CSVDataSet)...            data_catalog.py:343
                        INFO     Running node: Get-Current-Symbol: get_symbol([uni_crypto_features_data]) ->     node.py:327
                                [symbol]
                        INFO     Saving data to 'symbol' (MemoryDataSet)...                              data_catalog.py:382
                        INFO     Completed 1 out of 5 tasks                                          sequential_runner.py:85
                        INFO     Loading data from 'uni_crypto_features_data' (CSVDataSet)...            data_catalog.py:343
                        INFO     Running node: Train-and-Test-Data-Split:                                        node.py:327
                                train_test_split([uni_crypto_features_data]) -> [train_features,test_features]
    [02/07/23 13:28:08] INFO     Saving data to 'train_features' (MemoryDataSet)...                      data_catalog.py:382
                        INFO     Saving data to 'test_features' (MemoryDataSet)...                       data_catalog.py:382
                        INFO     Completed 2 out of 5 tasks                                          sequential_runner.py:85
                        INFO     Loading data from 'train_features' (MemoryDataSet)...                   data_catalog.py:343
                        INFO     Running node: Model-Training: train_model([train_features]) -> [model]          node.py:327
                        INFO     Saving data to 'model' (MemoryDataSet)...                               data_catalog.py:382
                        INFO     Completed 3 out of 5 tasks                                          sequential_runner.py:85
                        INFO     Loading data from 'model' (MemoryDataSet)...                            data_catalog.py:343
                        INFO     Loading data from 'test_features' (MemoryDataSet)...                    data_catalog.py:343
                        INFO     Running node: Model-Evaluation: evaluate_model([model,test_features]) ->        node.py:327
                                [y,y_pred,mse]
                        INFO     Saving data to 'y' (MemoryDataSet)...                                   data_catalog.py:382
                        INFO     Saving data to 'y_pred' (MemoryDataSet)...                              data_catalog.py:382
                        INFO     Saving data to 'mse' (MemoryDataSet)...                                 data_catalog.py:382
                        INFO     Completed 4 out of 5 tasks                                          sequential_runner.py:85
                        INFO     Loading data from 'symbol' (MemoryDataSet)...                           data_catalog.py:343
                        INFO     Loading data from 'y' (MemoryDataSet)...                                data_catalog.py:343
                        INFO     Loading data from 'y_pred' (MemoryDataSet)...                           data_catalog.py:343
                        INFO     Loading data from 'mse' (MemoryDataSet)...                              data_catalog.py:343
                        INFO     Running node: Display-Model-Evaluation-Metrics:                                 node.py:327
                                plot_metric([symbol,y,y_pred,mse]) -> None


    🤒 Mean Square Error (MSE) 0.109%


                        close_t1  close_pred_t1
    Timestamp
    2022-11-01 00:00:00    6.9463       6.948840
    2022-11-01 00:15:00    6.9716       6.970235
    2022-11-01 00:30:00    6.9570       6.957893
    2022-11-01 00:45:00    6.9723       6.971893
    2022-11-01 01:00:00    6.9933       6.991907
    ...                       ...            ...
    2022-12-31 22:45:00    5.1605       5.161068
    2022-12-31 23:00:00    5.1687       5.169422
    2022-12-31 23:15:00    5.1749       5.174875
    2022-12-31 23:30:00    5.1660       5.166717
    2022-12-31 23:45:00    5.1660            NaN

    [5554 rows x 2 columns]
                        INFO     Completed 5 out of 5 tasks                                          sequential_runner.py:85
                        INFO     Pipeline execution completed successfully.                                     runner.py:90
</details>


**Run the Kedo pipeline visualization from the command line**:

```bash
$ cd using_chatgpt_kedro_streamlit_app
$ kedro viz
```

You should see this displayed in a browser window:

![Pipeline Visualization](https://raw.githubusercontent.com/asehmi/using_chatgpt_kedro_streamlit_app/main/images/kedro_viz.png)

---

⭐ If you enjoyed this app and learned something, please consider starring its repository.

Many thanks!

Arvindra

---

## Disclaimer

**_This application is a demo of Kedro and Streamlit concepts and the results should not be taken seriously! The Linear Regression model is highly simplistic._**

- All investments involve risk, and the past performance of a crypto-currency, security, industry, sector, market, financial product, trading strategy, or individual’s trading does not guarantee future results or returns.
- Investors are fully responsible for any investment decisions they make. Such decisions should be based solely on an evaluation of their financial circumstances, investment objectives, risk tolerance, and liquidity needs.
- The information you derive from the outputs of this application do not constitute investment advice. I will not accept liability for any loss or damage, including without limitation any loss of profit, which may arise directly or indirectly from use of or reliance on such information.

---