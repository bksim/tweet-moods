from flask import Flask, render_template, request
import tweet_analyzer

app = Flask(__name__)

# Note: We don't need to call run() since our application is embedded within
# the App Engine WSGI application server.

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/result', methods=['GET'])
def result():
	query = str(request.args['query'])
	data = tweet_analyzer.get_sentiment_data(query, 'sentiment140')
	dates_list, percents_list = zip(*sorted(data.items(), key=lambda x:x[0]))

	month_list = [d.month for d in dates_list]
	day_list = [d.day for d in dates_list]
	day_list = day_list[-7:len(day_list)]

	percents_list = [100*d for d in percents_list]
	percents_list = percents_list[-7:len(percents_list)]

	neg = [str(100.0-d) for d in percents_list]
	percents_list = [str(d) for d in percents_list]
	return render_template('result.html', monthlist=month_list, daylist=day_list, positive_info=percents_list, negative_info=neg, query=query)

@app.errorhandler(404)
def page_not_found(e):
    """Return a custom 404 error."""
    return 'Sorry, nothing at this URL.', 404