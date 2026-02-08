import { useState } from 'react'
import {useDropzone} from 'react-dropzone';
import type { ImageFile } from '../ImageFile';
import imageUpload from '../assets/image-upload.png';

type Props = {
    setImages: React.Dispatch<React.SetStateAction<ImageFile[]>>;
};

function ImageUpload({ setImages }: Props) {
    const [error, setError] = useState<string>("");
    const MAX_SIZE = 500 * 1024; // 500 KB
    
    const { getRootProps, getInputProps, open } = useDropzone({
        accept: {
            "image/jpeg": [],
            "image/png": [],
            "image/webp": [],
            "image/gif": []
        },
        maxFiles: 100,
        noClick: true,
        onDrop: (acceptedFiles) => {
            const validFiles: ImageFile[] = [];
            const rejectedFiles: File[] = [];

            acceptedFiles.forEach(file => {
                if (file.size > MAX_SIZE) {
                    rejectedFiles.push(file);
                } else {
                    validFiles.push({
                        file,
                        preview: URL.createObjectURL(file),
                    });
                }
            });

            if (rejectedFiles.length > 0) {
                setError(
                    `Image(s) were too large. Max allowed size is 500 KB.`
                );
            }

            setImages(validFiles);
        },
    });

    return (
        <>
            <div className="image-upload-container">
                <div {...getRootProps({className: 'dropzone'})} className="upload">
                    <input {...getInputProps({
                        'aria-label': 'Upload an image to classify as fire, smoke, or non-fire'
                    })} />
                    <img src={imageUpload} alt="" width={50} />
                    <p>Drag and drop image(s) here</p>
                    <p>- or -</p>
                    <button type="button" onClick={open}>
                        Open File Dialog
                    </button>

                    {error && <p className="error">{error}</p>}
                </div>
            </div>
        </>
    )
}

export default ImageUpload
