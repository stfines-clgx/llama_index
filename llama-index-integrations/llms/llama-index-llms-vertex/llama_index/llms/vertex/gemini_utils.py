import base64
from typing import Any, Dict, Union, Optional
from vertexai.generative_models import FunctionDeclaration, Tool, Part, Content, GenerativeModel, Image
from llama_index.core.llms import ChatMessage, MessageRole
from vertexai.generative_models._generative_models import SafetySettingsType


def is_gemini_model(model: str) -> bool:
    return model.startswith("gemini")


def create_gemini_client(
    model: str, safety_settings: Optional[SafetySettingsType]
) -> Any:
    return GenerativeModel(model_name=model, safety_settings=safety_settings)


def convert_chat_message_to_gemini_content(
    message: ChatMessage, is_history: bool = True
) -> Any:


    def _convert_gemini_part_to_prompt(part: Union[str, Dict]) -> Part:

        if isinstance(part, str):
            return Part.from_text(part)

        if not isinstance(part, Dict):
            raise ValueError(
                f"Message's content is expected to be a dict, got {type(part)}!"
            )
        if part["type"] == "text":
            return Part.from_text(part["text"])
        elif part["type"] == "image_url":
            return _create_image_part(part)
        else:
            raise ValueError(
                f"Only text and image parts are supported!"
            )

    def _add_tool_calls(message: ChatMessage) -> list[Part]:
        _parts = list()
        if (("tool_calls" in message.additional_kwargs and len(message.additional_kwargs["tool_calls"]) > 0)
                or message.role == MessageRole.TOOL)\
                or (message.content == "" and "tool_calls" in message.additional_kwargs):
            tool_calls = message.additional_kwargs["tool_calls"]
            # function_decls = [
            #     FunctionDeclaration.from_func(tool_call) for tool_call in tool_calls]
            tools = [Part.from_function_response(tool_call.name, tool_call.args) for tool_call in tool_calls]

            _parts.extend(tools)

        return _parts

    def _create_image_part(part):
        path = part["image_url"]
        image_part = None
        if path.startswith("gs://"):
            image_part = Part.from_uri(path, mime_type="image/jpeg" if path.endswith(
                ".jpg") else "image/png" if path.endswith(
                ".png") else "image/gif")  # noqa: E5
        elif path.startswith("data:image/jpeg;base64,"):
            image = Image.from_bytes(base64.b64decode(path[23:]))
            image_part = Part.from_image(image)
        else:
            image = Image.load_from_file(path)
            image_part = Part.from_image(image)

        return image_part

    # Convert ChatMessage to GenerativeModel's Content type.
    parts = list()
    if message.content is not None and len(message.content) > 0\
            or (message.content =="" and "tool_calls" in message.additional_kwargs):
        parts.extend(_add_tool_calls(message))
    else:
        raw_content = message.content

        if raw_content is None:
            raw_content = [""]
        elif isinstance(raw_content, str):
            raw_content = [raw_content]
        else:
            raise ValueError(f"Unable to understand message content of type {type(raw_content)}!")

        parts.extend([_convert_gemini_part_to_prompt(r) for r in raw_content])

    if is_history:
        return Content(
            role="user" if message.role == MessageRole.USER else "model",
            parts=parts,
        )
    else:
        return Content(
            role=message.role,
            parts=parts, )



