class Message:
    def __init__(self, raw_message):
        self.id = raw_message.get("id")
        self.from_ = raw_message.get("from")
        self.type = raw_message.get("type")

        self.text = None
        self.location = None
        self.audio = None
        self.image = None
        self.interactive = None

        if self.type == "text":
            self.text = raw_message.get("text", {}).get("body")

        if self.type == "location":
            self.location = raw_message.get("location")

        if self.type == "audio":
            audio = raw_message.get("audio", {})
            self.audio = {
                "id": audio.get("id"),
                "mimeType": audio.get("mime_type"),
                "isVoice": audio.get("voice") is True
            }

        if self.type == "image":
            image = raw_message.get("image", {})
            self.image = {
                "id": image.get("id"),
                "mimeType": image.get("mime_type")
            }

        if self.type == "interactive":
            self.interactive = raw_message.get("interactive")

    def get_interaction(self):
        if self.type != "interactive" or not self.interactive:
            return None

        if self.interactive.get("type") == "button_reply":
            reply = self.interactive.get("button_reply", {})
            return {
                "kind": "BUTTON",
                "id": reply.get("id"),
                "title": reply.get("title")
            }

        if self.interactive.get("type") == "list_reply":
            reply = self.interactive.get("list_reply", {})
            return {
                "kind": "LIST",
                "id": reply.get("id"),
                "title": reply.get("title")
            }

        return None