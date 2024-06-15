import launch

if not launch.is_installed("lxml_html_clean"):
    launch.run_pip("install lxml_html_clean==0.1.1", "requirements for SD Model Preview XD")