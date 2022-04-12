# IBL spike web viewer

![Screenshot](https://user-images.githubusercontent.com/1942359/162991532-63d07f8c-06fa-4375-9305-e205a8d0384e.png)

### Install notes (to be completed)

* Tested on Ubuntu 20.04
* Create a Python virtual env
* `pip install -r requirements.txt`
* Build and install datoviz-distributed
* For deployment, `sudo nano /etc/systemd/system/flaskapp.service` and put:

```
[Unit]
Description=Flask app
After=network.target

[Service]
User=ubuntu
Group=www-data
WorkingDirectory=/home/ubuntu/dvzproto/distributed/ibl
Environment="PATH=/home/ubuntu/dataovizenv/bin"
Environment="LD_LIBRARY_PATH=/home/ubuntu/datoviz-distributed/build"
ExecStart=sudo LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/ubuntu/datoviz-distributed/build /home/ubuntu/datovizenv/bin/python flaskapp.py --port 80
#/home/ubuntu/datovizenv/bin/gunicorn --bind unix:flaskapp.sock --worker-class eventlet -w 1 -b 0.0.0.0:80 wsgi:app

[Install]
WantedBy=multi-user.target
```
