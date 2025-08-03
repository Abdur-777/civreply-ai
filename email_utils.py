import yagmail
import os

def send_civreply_email(
    user_email, question, answer, pdf_paths, 
    gmail_user="civreplywyndham@gmail.com", 
    gmail_app_password="lcjbssomhwkqalbm"   
):
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
            # This assumes you have a public URL to serve your PDFs, or just use filenames if local.
            pdf_url = f"https://yourwebsite.com/docs/wyndham/{os.path.basename(pdf)}"
            body += f'<a href="{pdf_url}">{os.path.basename(pdf)}</a><br>'
        body += "<br>"

    body += """
    If you have further questions, please reply to this email.<br><br>
    Best regards,<br>
    The CivReply AI Team<br>
    Wyndham City Council<br>
    """

    # Compose and send email
    yag = yagmail.SMTP(gmail_user, gmail_app_password)
    yag.send(to=user_email, subject=subject, contents=body)
    print(f"✅ Auto-reply sent to {user_email}")
