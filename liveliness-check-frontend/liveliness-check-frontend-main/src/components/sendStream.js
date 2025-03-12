import React, { useRef, useEffect, useState } from "react";
import io from "socket.io-client";
import { ThreeDots } from "react-loader-spinner";
import "./sendStreamMain.css";
import { RxCrossCircled } from "react-icons/rx";
import CustomAlertBox from "./customAlertBox";
import "react-responsive-modal/styles.css";
import { Modal } from "react-responsive-modal";
import gif from "../assets/left.gif"

const SendStream = () => {
  const videoRef = useRef(null);
  const videoRefForPhotoCapture = useRef(null);
  const [socket, setSocket] = useState(null);
  const [receivedFrame, setReceivedFrame] = useState("");
  const [sentFramesCount, setSentFramesCount] = useState(0);
  const [receivedFramesCount, setReceivedFramesCount] = useState(0);
  const [loading, setLoading] = useState(false);
  const [imageUploadOverlay, setImageUploadOverlay] = useState(false);
  const [selectedImage, setSelectedImage] = useState('');
  const [name, setName] = useState("");
  const [takePhoto, setTakePhoto] = useState(false);
  const [capturedPhoto, setCapturedPhoto] = useState("");
  const [alertMessage, setAlertMessage] = useState('');
  let   startLivelinessDetectionWith = "" ;
  const serverIP = process.env.REACT_APP_SERVER_IP ;
  const serverPort = process.env.REACT_APP_SERVER_PORT ;
  const [isLivelinessGateOpen, setIsLivelinessGateOpen] =  useState(true);

  let msg = new SpeechSynthesisUtterance()

  let longText =  "Your long text here. ".repeat(500);
  useEffect(() => {
    try {
      if (socket) {
        socket.on("connect", onConnect);
        socket.on("disconnect", onDisconnect);
        socket.on("response_from_server", onResponseReceived);
        socket.on("test_complete", handleTestComplete);
        socket.on("request_frame", handleSendFrame);
        socket.on("sound_instructions", handleSoundInstructions);
      }
      return cleanupSocket;
    } catch (error) {
      console.error("Error during socket setup:", error);
      // setAlertMessage("Error during socket setup. Please try again.");
      setAlertMessage("Something went wrong . Please try again.");
    }
  }, [socket]);

  const cleanupSocket = () => {
    console.log("socket clean up worked");
    try {
      if (socket) {
        console.log("Cleaning up socket");
        socket.off("connect", onConnect);
        socket.off("disconnect", onDisconnect);
        socket.off("response_from_server", onResponseReceived);
        socket.off("test_complete", handleTestComplete);
        socket.off("request_frame", handleSendFrame);
        socket.off("sound_instructions", handleSoundInstructions);

        if (socket.connected) {
          // Check if the socket is still open before closing
          socket.disconnect();
        }
        setReceivedFramesCount(0);
        setSentFramesCount(0);
        setSocket(null);
      }
    } catch (error) {
      console.error("Error during socket cleanup:", error);
      // setAlertMessage("Error during socket cleanup. Please try again.");
      setAlertMessage("Something went wrong. Please try again.");
      
    }
  };

  const onConnect = () => {
    console.log("Connected to server");
  };

  const onDisconnect = () => {
    console.log("Disconnected from server");
  };

  const onResponseReceived = (frame) => {
    try {
      setLoading(false);
      // console.log("frame received,", frame);
      setReceivedFrame(frame);
      setReceivedFramesCount((prevCount) => prevCount + 1);
    } catch (error) {
      setLoading(false);
      console.error("Error handling response:", error);
      // setAlertMessage("Error handling response. Please try again.");
      setAlertMessage("Something went wrong. Please try again.");
    }
  };

  const handleTestComplete = () => {
    try {
      console.log("Test complete event received. Turning off camera.");
      handleCameraOff();
      cleanupSocket();
    } catch (error) {
      console.error("Error during test completion:", error);
      setAlertMessage("Something went wrong. Please try again.");
    }
  };

  const handleCameraOff = () => {
    try {
      // Stop the camera stream
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject;
        const tracks = stream.getTracks();

        tracks.forEach((track) => {
          track.stop();
        });

        // Clear the video source object
        videoRef.current.srcObject = null;
      }
    } catch (err) {
      console.error("Error in stoping camera", err);
    }
  };

  const handleSendFrame = () => {
    try {
      console.log("frame requested");
      captureFrame();
    } catch (error) {
      setLoading(false);
      console.error("Error handling send frame:", error);
      setAlertMessage("Something went wrong. Please try again.");
    }
  };

  const startCamera = async () => {
    try {
      console.log("camera start 111111111111111111");
      setLoading(true);
      console.log(`http://${serverIP}:${serverPort}`)
      const newSocket = io(`http://${serverIP}:${serverPort}`, {
        reconnection: false, // Disable automatic reconnection
        reconnectionAttempts: 0, // Set the number of reconnection attempts to 0
      });

      // Add a catch block for handling connection errors
      newSocket.on("connect", () => {
        try {
          setSocket(newSocket);
          console.log("Connected successfully");
          performActionAfterConnection();
        } catch (err) {
          console.log("error in connect event listner:", err);
          // setAlertMessage("something went wrong. Please try again later.");
          setAlertMessage("Something went wrong. Please try again.");
        }
      });

      newSocket.on("connect_error", (error) => {
        console.error("Error connecting to the server:", error);
        // handleCameraOff();
        // setAlertMessage("Error connecting to the server. Please try again later.");
        // <CustomAlertBox message = "Please select an image and enter a name." ></CustomAlertBox>
        setAlertMessage("Error connecting to the server. Please try again later.");
        setLoading(false);
      });

      const performActionAfterConnection = async () => {
        try {
          console.log(
            "scoketvalue inside performActionAfterConnection:",
            socket
          );
          const stream = await navigator.mediaDevices.getUserMedia({
            video: true,
          });
          // If the camera access is successful (stream is truthy)
          if (stream) {
            if (videoRef.current) {
              videoRef.current.srcObject = stream;
              // Wait for a short duration before emitting 'camera_started'
              setTimeout(() => {
                console.log("Camera started");
                try {
                  // if(startLivelinessDetectionWith === "blink"){
                    console.log("startLivelinessDetectionWith1111111111111", startLivelinessDetectionWith)
                    newSocket.emit("camera_started", startLivelinessDetectionWith );
                  // }else if (startLivelinessDetectionWith === "face movement"){
                  //   newSocket.emit("camera_started for face movement");
                  // }
                } catch (err) {
                  setLoading(false);
                  console.log("error in emitting camera_started event:", err);
                  // setAlertMessage(
                  //   "Error in connection with server. Please try again."
                  // );
                  setAlertMessage("Error in connection with server. Please try again.");

                }
              }, 2000);
            } else {
              // Handle the case where videoRef.current is not available
              setLoading(false);
              console.error(
                "Error accessing camera: videoRef.current is not available"
              );
              // setAlertMessage(
              //   "Error accessing camera. Please check your camera and try again."
              // );
              setAlertMessage("Error accessing camera. Please check your camera and try again.")
            }
          } else {
            // Handle the case where stream is not truthy
            setLoading(false);
            console.error(
              "Error accessing camera: getUserMedia did not return a valid stream"
            );
            // setAlertMessage(
            //   "Error accessing camera. Please check your camera and try again."
            // );
            setAlertMessage( "Error accessing camera. Please check your camera and try again.")
          }
        } catch (error) {
          setLoading(false);
          console.error(
            "Error accessing camera: navigator.mediaDevices.getUserMedia not working properly "
          );
          // setAlertMessage(
          //   "Error accessing camera. Please check your camera and try again."
          // );
          setAlertMessage( "Error accessing camera. Please check your camera and try again.")
        }
      };
    } catch (error) {
      setLoading(false);
      console.error("Error during connection with server:", error);
      // Handle other errors during camera setup
      // setAlertMessage("Error in connecting with the server. Please try again.");
      setAlertMessage( "Error in connecting with the server. Please try again.")
    }
  };

  // const createConnection = async ()=> {
  //     const newSocket = io('http://192.168.29.86:3005', {
  //             reconnection: false,  // Disable automatic reconnection
  //             reconnectionAttempts: 0,  // Set the number of reconnection attempts to 0
  //         });

  //         // Add a catch block for handling connection errors
  //         newSocket.on('connect_error', (error) => {
  //             console.error('Error connecting to the server:', error);
  //             handleCameraOff();
  //             setAlertMessage('Error connecting to the server. Please try again later.');
  //             setLoading(false);
  //             return false;
  //         });

  //         console.log("after newSocket.on executed")
  //         setSocket(newSocket);
  // }

  const sendDataToServer = (imageData) => {
    try {
      // console.log("imageData in sendDataToServer function", imageData);
      if (socket) {
        socket.emit("frame_from_client", imageData);
        setSentFramesCount((prevCount) => prevCount + 1);
        // console.log("frame sent");
      }
    } catch (error) {
      setLoading(false);
      console.error("Error sending data to server:", error);
      // setAlertMessage("Error sending data to server. Please try again.");
      setAlertMessage("Something went wrong. Please try again.");
    }
  };

  const captureFrame = () => {
    try {
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");

      if (videoRef.current && socket) {
        canvas.width = videoRef.current.videoWidth;
        canvas.height = videoRef.current.videoHeight;

        context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);

        const imageData = canvas.toDataURL("image/jpeg");
        sendDataToServer(imageData);
      }
    } catch (error) {
      setLoading(false);
      console.error("Error capturing frame:", error);
      // setAlertMessage("Error capturing frame. Please try again.");
      setAlertMessage("Something went wrong. Please try again.");
    }
  };

  const handleBlinkStart = () =>{
    // setStartLivelinessDetectionWith("blink")
    
    // if(!isLivelinessGateOpen){
    //   return;
    // }

    // setIsLivelinessGateOpen(true);

    startLivelinessDetectionWith = "blink";
    startCamera();
  }

  const handleFaceMovementDetection = () =>{
    // setStartLivelinessDetectionWith("face movement")
    startLivelinessDetectionWith = "face movement"
    startCamera();
  }

  const handleStartButtonClick = () => {
    startCamera();
  };

  const handleImageUpload = () => {
    setImageUploadOverlay(true);
  };

  const handleCloseClick = () => {
    setImageUploadOverlay(false);
    setCapturedPhoto('')
    setSelectedImage(false);
  };

  const handleImageChange = (event) => {
    try {
      setCapturedPhoto('')
      const file = event.target.files[0];
      if (file) {
        const reader = new FileReader();
        reader.onloadend = () => {
          setSelectedImage(reader.result); // Save base64-encoded image data
        };
        reader.readAsDataURL(file);
      }
    } catch (err) {
      console.log("error in choosing image file :", err);
      // setAlertMessage("Error in selecting image for upload, Please try again");
      setAlertMessage("Error in selecting image for upload, Please try again");
    }
  };

  const handleUpload = async () => {
    try {
      setLoading(true);
      if (selectedImage && name) {
        try {
          // Make an API request to send the base64 image data and name to the server
          const response = await fetch(
            `http://${serverIP}:${serverPort}/upload_image`,
            {
              method: "POST",
              headers: {
                "Content-Type": "application/json",
              },
              body: JSON.stringify({ name: name, image: selectedImage }),
            }
          );

          // Handle the response from the server as needed
          const data = await response.json();
          console.log("Server Response:", data);


          if (data.result === "Passed") {
            // setAlertMessage(data.upload_message);
            setAlertMessage(data.upload_message);
            setSelectedImage('');
            handleCloseClick();
            setName('')
            setCapturedPhoto('')
          } else {
            // setAlertMessage(data.upload_message);
            setAlertMessage(data.upload_message);
            setSelectedImage('');
            setCapturedPhoto('')
            console.log(selectedImage);
          }
        } catch (error) {
          console.error("Error uploading data.. api not working properly:", error);
          // setAlertMessage("Error uploading data, Check your connection and try again");
          setAlertMessage("Error uploading data, Check your connection and try again");
        }
      } else if(!name && !selectedImage) {
        // setAlertMessage("Please select an image and enter a name.");
        setAlertMessage("Please select an image and enter a name.");
      } else if(!selectedImage){
        // setAlertMessage("Please select an image ");
        setAlertMessage("Please select an image");
      }else{
        // setAlertMessage("Please enter  name.");
        setAlertMessage("Please enter  name.");
      }
    } catch (err) {
      console.log("error in hitting api to send image to the server:", err);
      // setAlertMessage("server error... plaease try again later");
      setAlertMessage("something went wrong, Please try again");
    } finally{
      setLoading(false);
    }
  };

  const handleNameChange = (event) => {
    try {
      setName(event.target.value);
    } catch (err) {
      console.log("error in setting the name to the state variable name:", err);
      // setAlertMessage("something went wrong... plaease try again");
      setAlertMessage("Something went wrong. Please try again.");
    }
  };

  const StartPhotoCaptureCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
      });
      console.log("camera started to capture image");
      if (stream) {
        if (videoRefForPhotoCapture.current) {
          videoRefForPhotoCapture.current.srcObject = stream;
        } else {
          // Handle the case where videoRef.current is not available
          setLoading(false);
          console.error(
            "Error accessing camera: videoRef.current is not available"
          );
          setAlertMessage(
            "Error accessing camera. Please check your camera and try again."
          );
        }
      } else {
        // Handle the case where stream is not truthy
        setLoading(false);
        console.error(
          "Error accessing camera: getUserMedia did not return a valid stream"
        );
        setAlertMessage(
          "Error accessing camera. Please check your camera and try again."
        );
      }
    } catch (error) {
      setLoading(false);
      console.error(
        "Error accessing camera: navigator.mediaDevices.getUserMedia not working properly "
      );
      setAlertMessage(
        "Error accessing camera. Please check your camera and try again."
      );
    }
  };

  const capturePhoto = () => {
    try {
      console.log("capturing Photo");
      const canvas = document.createElement("canvas");
      const context = canvas.getContext("2d");

      if (videoRefForPhotoCapture.current) {
        canvas.width = videoRefForPhotoCapture.current.videoWidth;
        canvas.height = videoRefForPhotoCapture.current.videoHeight;

        context.drawImage(
          videoRefForPhotoCapture.current,
          0,
          0,
          canvas.width,
          canvas.height
        );

        const imageData = canvas.toDataURL("image/jpeg");

        setCapturedPhoto(imageData);

        handleCameraOffForPhotoCapture();

        //    // Create a new image element and set its source to the captured image data
        // const capturedImage = new Image();
        // capturedImage.src = imageData;

        // // Append the image element to the same container as the video
        // videoRefForPhotoCapture.current.parentNode.appendChild(capturedImage);
      }
    } catch (error) {
      setLoading(false);
      console.error("Error capturing frame:", error);
      setAlertMessage("Error capturing frame. Please try again.");
    }
  };

  const handleTakePhotoClick = () => {
    setTakePhoto(true);
    setCapturedPhoto('');
    StartPhotoCaptureCamera();
  };

  const handlePhotoClick = () => {
    capturePhoto();
  };

  const handleCameraOffForPhotoCapture = () => {
    try {
      // Stop the camera stream
      console.log("closing the camera");
      if (
        videoRefForPhotoCapture.current &&
        videoRefForPhotoCapture.current.srcObject
      ) {
        const stream = videoRefForPhotoCapture.current.srcObject;
        const tracks = stream.getTracks();

        tracks.forEach((track) => {
          track.stop();
        });

        // Clear the video source object
        videoRefForPhotoCapture.current.srcObject = null;
      }
    } catch (err) {
      console.error("Error in stoping camera", err);
    }
  };

  const handleSubmitPhoto = () => {
    try{
    setSelectedImage(capturedPhoto)
    // setCapturedPhoto('')
    handleCameraOffForPhotoCapture();
    setTakePhoto(false);
    }catch(err){
      console.log("error in handle submit", err)
    }
  };

  const handleSoundInstructions = (inputInstruction) => {
    // msg.text = instruction
    console.log("receiving sound instructions", );
    // window.speechSynthesis.speak(msg)
    const synth = window.speechSynthesis;
    const utterance = new SpeechSynthesisUtterance(inputInstruction);
    synth.speak(utterance);
  }

  msg.onerror = function(event) {
    console.error('Speech synthesis error:', event.error);
  };

  return (
    <div className="sendStreamContainer">
        <div className="instruction-box">
        <div>
  If you are coming for the <strong style={{ fontWeight: '1000' }}>first time,</strong> then upload your image and start liveliness detection.
</div>
           <div style={{display:'flex', justifyContent:'center'}}>
            <button onClick={handleImageUpload} className="button">
              Upload Image
            </button>
           </div>
        </div>
      {/* <div className="receivedFramesVideoDiv"> */}
        {receivedFrame && (
          <img
            src={`${receivedFrame}`}
            alt="Processed Frame"
            className="receivedFramesVideo"
          />
        )}

      {/* </div> */}
      <div className="start-liveliness-buttons" >
        <div className="leftButtonDiv">
         <button className="button"  onClick={handleBlinkStart}>Blink Liveliness Detection</button>
        </div>
        <div className="rightButtonDiv">
         <button className="button"  onClick={handleFaceMovementDetection}>Face Movement Liveliness Detection</button>
         <img src={gif} alt="Your GIF Description" style={{ width: '50px', height: 'auto' }} />
        </div>
      </div>
      <div className="instruction-box-two">
       <div className="instruction-box-blink">
        <strong style={{fontSize:'1.3rem', fontWeight:'2000'}}>Instructions for blink eye:</strong>
        <div>
          1. It is recommended not to wear glasses during this activity for optimal results.<br/>
          2. Maintain focus and avoid excessive movement during the exercise. <br/>
          3. Ensure you are in a well-lit area to enhance visibility. <br/>
          4. Maintain eye level with the camera .
        </div>
       </div>
       <div className="instruction-box-face-movement">
        <strong style={{fontSize:'1.3rem', fontWeight:'2000'}}>Instructions for face movement:</strong>
        <div>
          1. It is recommended not to wear glasses during this activity for optimal results.<br/>
          2. Move your face to the left and right fully for liveliness detection. <br/>
          3. Ensure you are in a well-lit area to enhance visibility. <br/>
          4. Maintain eye level with the camera .<br/>
          5. Remain in the frame until liveliness detection is complete.
        </div>
       </div>
      </div>
      {/* <p>
        Sent Frames: {sentFramesCount} Received Frames: {receivedFramesCount}{" "}
      </p> */}
      <div className="cleint-side-video">
      <video ref={videoRef} autoPlay playsInline className="video" width= '320px' height = '240px'  style={{ }} />
      </div>

      {imageUploadOverlay && (
        <Modal open={imageUploadOverlay} onClose={()=>{}} center = {true}  closeIcon={true}    styles={{
          modal: {
            padding:0,
            border: "2px solid #4fa94d",
            borderRadius: "10px",
            overlay: {
              backgroundColor: "transparent", // Make the background fully transparent
            },
           },
         }}
        >
        <div className="image-upload-overlay">
          <div style={{position:'absolute', right: 5, top: 5 }}> 
            <RxCrossCircled size={30} onClick={handleCloseClick} />
          </div>
          <h2 style={{}}>Upload Image</h2>
          <div className="image-input-box">
            <input
              type="file"
              accept="image/*"
              onChange={handleImageChange}
              style={{
                width: '80%',  // Set the width to 100% to make it responsive
                padding: '10px', // Add padding as needed
                minWidth:"20px"
              }}
            />
            <strong>OR</strong>
            <button onClick={handleTakePhotoClick} style={{margin:'10px'}}>Take Photo</button>
            {selectedImage && <p style={{ fontSize:'1.2rem', color:'green' , fontWeight:'bolder', textAlign:'center', margin:'0px', padding:'0px'}}>Photo captured successfully, Enter your name and Upload.  </p>}
          </div>
          <input
            type="text"
            placeholder="Enter name"
            value={name}
            onChange={handleNameChange}
            style={{minWidth:'200px', width:'55%'}}
          />
          <button onClick={handleUpload} className="button" style={{}}>
            Upload
          </button>
          <div className="instruction-box-imageData">
            <strong>Instructions:</strong>
            <p>
             1. Ensure the image size is within the range of 220kb to 5MB.<br/>
             2. The photo must be passport size.<br/>
             3. Choose a well-lit area for the background.<br/>
             4. There should be only one person in the frame.<br/>
             5. Ensure that the full face is captured within the frame.<br/>
            </p>
          {/* </div>
          <div>{longText} */}
          </div>
          </div>
        </Modal>
      )}

      {takePhoto && (
        <Modal open={takePhoto} center = {true} onClose={()=>{}} closeIcon = {true} styles={{
              modal: {
                border: "2px solid #4fa94d",
                borderRadius: "10px",
                height:"90%",
                paddingTop:'40px',
                // width:"50%",
                // minWidth:'350px',
              }
            }}  >
            <div className="photoCaptureOverlay">
              <div style={{ position: "absolute", right: 4, top: 4 }}>
               <RxCrossCircled size={30} onClick={()=>{setTakePhoto(false); handleCameraOffForPhotoCapture();}} />
              </div>
              {(videoRefForPhotoCapture && !capturedPhoto ) && (
                <video
                  ref={videoRefForPhotoCapture}
                  autoPlay
                  playsInline
                  className="videoRefForPhotoCapture"
                  style={{border: "2px solid #4fa94d" }}
                />
              )}

              {capturedPhoto && (
                <img
                  src={`${capturedPhoto}`}
                  alt="Processed Frame"
                  className="videoRefForPhotoCapture"
                />
              )}
              <div
                style={{
                  display: "flex",
                  flexDirection: "row",
                  justifyContent: "space-between",
                  margin: "10px",
                }}
              >
                <div style={{ marginRight: "10px" }}>
                  <button className="button" onClick={handlePhotoClick}>Click Photo</button>
                </div>
                <div style={{ marginRight: "10px" }}>
                  <button onClick={handleSubmitPhoto} className="button">Submit Photo</button>
                </div>
                <div style={{ marginRight: "10px" }}>
                  <button
                    onClick={() => {
                      setCapturedPhoto("");
                      handleCameraOffForPhotoCapture();
                      StartPhotoCaptureCamera();
                    }}
                    className="button"
                  >
                    Capture Again
                  </button>
                </div>
              </div>
            </div>
        </Modal>
      )}

      {loading && (
        <div className="loaderOverlay">
          <ThreeDots
            height="80"
            width="80"
            radius="9"
            color="#4fa94d"
            ariaLabel="three-dots-loading"
            wrapperStyle={{}}
            wrapperClassName=""
            visible={true}
          />
        </div>
      )}

      {alertMessage && <CustomAlertBox message = {alertMessage} closeAlert = {()=>{setAlertMessage('')}} ></CustomAlertBox>}
    </div>
  );
};

export default SendStream;
