from flask import *

from LR_clssifier import predict

app=Flask(__name__)

@app.route("/", methods=['GET'])
def load():
    prediction = "thinking"
    text = ""
    if len(request.args) > 0:
        text = request.args.get('text').strip()
        if text != "":
            prediction = predict(text)
    return render_template("load.html", prediction = prediction, text = text)


if __name__=="__main__":
    app.run(debug=True)