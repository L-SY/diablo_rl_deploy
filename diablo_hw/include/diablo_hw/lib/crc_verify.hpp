/**
 * @file crc_verify.hpp
 * @brief CRC16 checksum calculation and varification
 */
#pragma once

#include <iostream>

namespace JOINT_CTRL{

/**
 * @brief verify CRC16 value of a received data packet
 * @return true if CRC value is correct
 * @note NON-API FUNCTION
 */
bool     verify_crc16(const void* data, size_t len);

/**
 * @brief calculate CRC16 value of a data packet
 * @param[in] len: shall equal to sizeof(packet) - 2 
 * @note NON-API FUNCTION
 */
uint16_t update_crc16(const void* data, size_t len);

}
