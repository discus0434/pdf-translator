from pathlib import Path

import numpy as np

from .ppocr_onnx.ppocr_onnx import PaddleOcrONNX


class _DictDotNotation(dict):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.__dict__ = self


class OCRModel:
    def __init__(self, model_root_dir: Path, device: str = "cuda") -> None:
        """
        Initialize OCR model.

        Parameters
        ----------
        model_root_dir : Path
            Path to the PaddleOCR model root directory.
        device : str, optional
            Device to use, by default "cuda"

        Raises
        ------
        FileNotFoundError
            If the model directory is not found.
        """
        self.paddleocr_parameters = self.__get_paddleocr_parameters(
            model_root_dir, device
        )
        self.paddleocr = PaddleOcrONNX(self.paddleocr_parameters)

    def __call__(self, image: np.ndarray) -> str:
        """
        Perform OCR on the image.

        Parameters
        ----------
        image : np.ndarray
            RGB image data

        Returns
        -------
        str
            OCR result
        """
        return self.paddleocr(image)

    def __get_paddleocr_parameters(
        self, model_root_dir: Path, device: str
    ) -> _DictDotNotation:
        """
        Get parameters for PaddleOCR.

        Parameters
        ----------
        model_root_dir : Path
            Path to the PaddleOCR model root directory.
        device : str
            Device to use.

        Returns
        -------
        _DictDotNotation
            PaddleOCR parameters.
        """
        paddleocr_parameters = _DictDotNotation()

        # params for prediction engine
        paddleocr_parameters.use_gpu = True if device == "cuda" else False

        # params for text detector
        paddleocr_parameters.det_algorithm = "DB"
        paddleocr_parameters.det_model_dir = str(
            model_root_dir / "en_PP-OCRv3_det_infer.onnx"
        )
        paddleocr_parameters.det_limit_side_len = 960
        paddleocr_parameters.det_limit_type = "max"
        paddleocr_parameters.det_box_type = "quad"

        # DB parmas
        paddleocr_parameters.det_db_thresh = 0.3
        paddleocr_parameters.det_db_box_thresh = 0.6
        paddleocr_parameters.det_db_unclip_ratio = 1.5
        paddleocr_parameters.max_batch_size = 10
        paddleocr_parameters.use_dilation = False
        paddleocr_parameters.det_db_score_mode = "fast"

        # params for text recognizer
        paddleocr_parameters.rec_algorithm = "SVTR_LCNet"
        paddleocr_parameters.rec_model_dir = str(
            model_root_dir / "en_PP-OCRv3_rec_infer.onnx"
        )
        paddleocr_parameters.rec_image_shape = "3, 48, 320"
        paddleocr_parameters.rec_batch_num = 6
        paddleocr_parameters.rec_char_dict_path = str(model_root_dir / "en_dict.txt")
        paddleocr_parameters.use_space_char = True
        paddleocr_parameters.drop_score = 0.5

        # params for text classifier
        paddleocr_parameters.use_angle_cls = False
        paddleocr_parameters.cls_model_dir = str(
            model_root_dir / "ch_ppocr_mobile_v2.0_cls_infer.onnx"
        )
        paddleocr_parameters.cls_image_shape = "3, 48, 192"
        paddleocr_parameters.label_list = ["0", "180"]
        paddleocr_parameters.cls_batch_num = 6
        paddleocr_parameters.cls_thresh = 0.9

        paddleocr_parameters.save_crop_res = False

        return paddleocr_parameters
