import React from 'react';
import { RxCrossCircled } from 'react-icons/rx';
import "react-responsive-modal/styles.css";
import { Modal } from "react-responsive-modal";

const CustomAlertBox = ({ message, closeAlert }) => {


  return (
 
    <Modal
    open={message}
    center={true}
    closeIcon={true}
    onClose={()=>{}}
    styles={{
      modal: {
        padding: 10,
        border: "2px solid #4fa94d",
        borderRadius: "10px",
      },
    }}
  >
 <div style={{ display: 'flex', flexDirection: 'column', justifyContent: "center", alignItems: 'center', borderRadius: "20px", minHeight: '150px',  margin: '0 auto', padding: '20px', minWidth:'250px'}}>
    <div style={{ position: "absolute", right: 4, top: 4 }}>
      <RxCrossCircled size={30} onClick={closeAlert} />
    </div>
    <p style={{ fontSize: '20px', marginBottom: '10px', fontWeight:'bolder'}}>Alert</p>
    <p style={{textAlign:'center'}}>{message}</p>
    <button className = 'button' onClick={closeAlert}>
      OK
    </button>
    </div>
  </Modal>
  );
}

export default CustomAlertBox;
