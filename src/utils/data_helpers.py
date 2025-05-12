import pandas as pd
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
from config.loader import load_config

def process_heat_load_data(df):
    """
    Process heat load data for visualization.
    
    Args:
        df (pd.DataFrame): Raw heat load data.
    
    Returns:
        pd.DataFrame: Processed DataFrame suitable for plotting.
    """
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df.set_index("timestamp", inplace=True)
    return df

def email_missing_variables(missing_vars, context_info=None):
    """
    Send an email to the project owner if variables are missing in the live API response.
    Args:
        missing_vars (list): List of missing variable names.
        context_info (str, optional): Additional context to include in the email.
    """
    config = load_config()
    # You should add a field like 'project_owner_email' in your setting.yaml
    owner_email = config.get('project_owner_email', None)
    if not owner_email:
        logging.error("Project owner email not configured in setting.yaml.")
        return
    subject = "[Evonith BF Webapp] Missing Variables in Live API Response"
    body = f"The following variables are missing in the live API response:\n\n" \
           f"{missing_vars}\n\n"
    if context_info:
        body += f"Context: {context_info}\n"
    body += "\nPlease investigate the data source or API.\n"
    # Configure your SMTP server here
    smtp_server = config.get('smtp_server', 'smtp.gmail.com')
    smtp_port = config.get('smtp_port', 587)
    smtp_user = config.get('smtp_user', None)
    smtp_password = config.get('smtp_password', None)
    if not smtp_user or not smtp_password:
        logging.error("SMTP credentials not configured in setting.yaml.")
        return
    msg = MIMEMultipart()
    msg['From'] = smtp_user
    msg['To'] = owner_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_user, smtp_password)
        server.sendmail(smtp_user, owner_email, msg.as_string())
        server.quit()
        logging.info(f"Missing variable email sent to {owner_email}.")
    except Exception as e:
        logging.error(f"Failed to send missing variable email: {e}")
