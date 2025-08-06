import yagmail
import os

def send_civreply_email(user_email, question, answer, pdf_paths):
    """
    Send a CivReply AI answer to a user's email using yagmail and a Gmail app password.
    PDF links can be attached (as URLs, not files).
    """
    gmail_user = "civreplywyndham@gmail.com"
    # NEVER hardcode your app password. Always use an environment variable!
    gmail_app_password = os.environ.get("CIVREPLY_GMAIL_APP_PASSWORD")
    if not gmail_app_password:
        raise ValueError("CIVREPLY_GMAIL_APP_PASSWORD env var is not set.")

    subject = "CivReply AI – Your Council Question Answered"
    body = f"""
    Hello,<br><br>
    Thank you for reaching out to Wyndham Council.<br><br>
    <b>Your question:</b><br>
    {question}<br><br>
    <b>Our answer:</b><br>
    {answer}<br><br>
    """

    if pdf_paths:
        body += "For more information, please see the following official document(s):<br>"
        for pdf in pdf_paths:
            pdf_url = f"https://yourwebsite.com/docs/wyndham/{os.path.basename(pdf)}"
            body += f'<a href="{pdf_url}">{os.path.basename(pdf)}</a><br>'
        body += "<br>"

    body += """
    If you have further questions, please reply to this email.<br><br>
    Best regards,<br>
    The CivReply AI Team<br>
    Wyndham City Council<br>
    """

    try:
        yag = yagmail.SMTP(gmail_user, gmail_app_password)
        yag.send(to=user_email, subject=subject, contents=body)
        print(f"✅ Auto-reply sent to {user_email}")
        return True
    except Exception as e:
        print(f"❌ Email failed: {e}")
        return False
