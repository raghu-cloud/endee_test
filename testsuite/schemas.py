from pydantic import BaseModel


class RunTestRequest(BaseModel):
    test_run_id: int

class UpdateLibraryVersionRequest(BaseModel):
    version_update_id: int