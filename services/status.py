class Status:
    def __init__(self, raw_status):
        self.message_id = raw_status.get("id")
        self.status = raw_status.get("status")
        self.recipient_phone_number = raw_status.get("recipient_id")