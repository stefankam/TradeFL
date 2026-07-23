"""Communication accounting by serialized payload size."""
from __future__ import annotations
import pickle
from typing import Any
class CommunicationCounter:
    def __init__(self)->None: self.uploads=[]; self.downloads=[]
    def _size(self,payload:Any)->int: return len(payload) if isinstance(payload,(bytes,bytearray)) else len(pickle.dumps(payload))
    def record_upload(self,payload:Any,label:str)->None: self.uploads.append((label,self._size(payload)))
    def record_download(self,payload:Any,label:str)->None: self.downloads.append((label,self._size(payload)))
    @property
    def total_uploaded_bytes(self)->int: return sum(x[1] for x in self.uploads)
    @property
    def total_downloaded_bytes(self)->int: return sum(x[1] for x in self.downloads)
