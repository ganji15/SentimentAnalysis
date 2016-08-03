# SentimentAnalysis

  'imdb.pkl' is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We use RNN architecture to solve this problem, such as lstm, blstm and special lstm. However, 'blstm' and 'lstm using meanpooling' gets the best results.


# Results
|           | lstm           | blstm         | mlstm      |          mblstm    |
| --------|:------------:|:------------:|:------------:|-----------:|
| train loss|0.333254 | 0.016042 |  0.248304  |  0.371451|
| valid acc |   83.81%  |  85.24% | 88.10%     |  89.52%  |
| test acc  |   76.27%  |  82.80% | 81.56%     |    79.02% |

# Reference
* [http://ai.stanford.edu/~amaas/data/sentiment/](http://ai.stanford.edu/~amaas/data/sentiment/)
* [http://deeplearning.net/tutorial/lstm.html](http://deeplearning.net/tutorial/lstm.html)


# Contacts
ganji15@mails.ucas.ac.cn
