class Server:
    def aggregate(self, updates:list[dict])->dict: return {'num_updates': len(updates)}
