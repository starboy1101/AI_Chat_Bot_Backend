REQUIREMENT_FIELDS = {
    "service_select": {
        "desc": "Type of service requested",
        "examples": [
            "audio porting",
            "audio optimization",
            "audio application development",
            "DSP optimization",
        ],
    },
    "Optimization_type": {
        "desc": "Type of optimization requested",
        "examples": [
            "design level optimization",
            "intrinsic optimization",
            "algorithm optimization",
        ],
    },
    "Porting_type": {
        "desc": "Porting related requirement or target",
        "examples": [
            "porting to Qualcomm",
            "DSP porting",
            "migration from TI to QCOM",
        ],
    },
    "DSP_Processor": {
        "desc": "Target DSP or processor",
        "examples": [
            "Qualcomm Hexagon",
            "HiFi 4",
            "HiFi 5",
            "TI C6000",
            "QCOM 855",
        ],
    },
    "TargetPlatform_1": {
        "desc": "Target platform or SoC family",
        "examples": [
            "Qualcomm platform",
            "TI platform",
            "Hexagon DSP",
        ],
    },
    "TargetPlatform_2": {
        "desc": "Specific chipset or hardware details",
        "examples": [
            "QCOM 845",
            "QCOM 855",
            "TI TDA2x",
            "Snapdragon 865",
        ],
    },
    "Audio_Params_1": {
        "desc": "PCM bit depth",
        "examples": ["16-bit", "24-bit", "32-bit"],
    },
    "Audio_Params_2": {
        "desc": "Sampling frequency",
        "examples": ["48 kHz", "96 kHz", "192 kHz"],
    },
    "Audio_Params_3": {
        "desc": "Audio channel format",
        "examples": ["mono", "stereo", "multi-channel"],
    },
}