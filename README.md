# Facial recognition API

This API is used for registering new faces to be recognized in the future, and for recognizing existing faces.
This can be used for multiple purposes, like a security system based on image recognition.

## Installation

# Deploying to AWS EC2

Log into your AWS account and create an EC2 instance (`t2.micro`), using the latest stable
Ubuntu Linux AMI.

[SSH into the instance](https://aws.amazon.com/blogs/compute/new-using-amazon-ec2-instance-connect-for-ssh-access-to-your-ec2-instances/) and run these commands to update the software repository and install
our dependencies.

```bash
sudo apt-get update
sudo apt install -y python3-pip nginx
```

Add the FastAPI configuration to NGINX's folder. Create a file called `fastapi_nginx` (like the one in this repository).

```bash
sudo vim /etc/nginx/sites-enabled/fastapi_nginx
```

And put this config into the file (replace the IP address with your EC2 instance's public IP):

```
server {
    listen 80;   
    server_name <YOUR_EC2_IP>;    
    location / {        
        proxy_pass http://127.0.0.1:8000;    
    }
}
```

Install all the dependencies on requirements.txt
```bash
python3 -m pip install -r requirements.txt
```

## Usage

Start NGINX.

```bash
sudo service nginx restart
```

Start FastAPI.

```bash
python3 -m uvicorn main:app
```

# Endpoint register-person
Use this endpoint to register the name, the photo and the status of a person.
The status can be used for acknowledge an specific person of a photo, for example, friend or family.

# Endpoint recognize-image
Use this endpoint to upload a photo and verify the emotion of the person and if this person is registered on the system.

## Accessing the API
If you use this API on EC2 an followed the configurations above, go on browser and access via: 
```
http://YOUR_EC2_IP
```
Dont forget to check de documentation for testing and learning how to use
```
http://YOUR_EC2_IP/docs/
```

If you`re using this API on localhost:
```
http://127.0.0.1:8000/docs/
```
## License

[MIT](https://choosealicense.com/licenses/mit/)