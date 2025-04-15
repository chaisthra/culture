from flask import Flask, render_template, request, redirect, url_for, make_response, session
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for sessions

# Dummy user data
users = {
    "user1@example.com": generate_password_hash("password123")
}

@app.route('/')
def index():
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        remember = request.form.get('remember')

        user_pass_hash = users.get(email)
        if user_pass_hash and check_password_hash(user_pass_hash, password):
            session['user'] = email
            resp = make_response(redirect(url_for('dashboard')))
            if remember:
                resp.set_cookie('userEmail', email, max_age=60*60*24*30)  # 30 days
            return resp
        else:
            return render_template('login.html', error="Invalid email or password")
    
    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        confirm = request.form['confirm']

        if email in users:
            return render_template('signup.html', error="Email already exists")
        if password != confirm:
            return render_template('signup.html', error="Passwords do not match")

        users[email] = generate_password_hash(password)
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'user' in session:
        return f"<h2>Welcome, {session['user']}!</h2><br><a href='/logout'>Logout</a>"
    return redirect(url_for('login'))

@app.route('/logout')
def logout():
    session.pop('user', None)
    resp = make_response(redirect(url_for('login')))
    resp.set_cookie('userEmail', '', expires=0)
    return resp

if __name__ == '__main__':
    app.run(debug=True)