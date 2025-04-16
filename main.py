from flask import Flask, request, make_response

app = Flask(__name__)

@app.route('/')
def index():
    username = request.cookies.get('username')
    if username:
        return f'Welcome back, {username}!'
    return 'Hello, Guest!'

@app.route('/set_cookie')
def set_cookie():
    resp = make_response("Cookie Set!")
    resp.set_cookie('username', 'JohnDoe')  # name, value
    return resp

@app.route('/delete_cookie')
def delete_cookie():
    resp = make_response("Cookie Deleted!")
    resp.delete_cookie('username')
    return resp

if __name__ == '__main__':
    app.run(debug=True)
