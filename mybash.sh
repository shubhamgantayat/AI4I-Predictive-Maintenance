git config user.email "sgantayat9@gmail.com"
git config user.name "Shubham Gantayat"
curl https://cli-assets.heroku.com/install.sh | sh
git add .
git commit -m "mlflow-commit"
heroku login
git push heroku main