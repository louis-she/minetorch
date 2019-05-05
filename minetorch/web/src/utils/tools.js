import { camelCase } from 'lodash'

/**
 * [获取url参数]
 * @param key
 * @param href
 */
const getQueryVariable = (key, href) => {
  let match = decodeURIComponent(href || window.location.href).match(new RegExp('[?&]' + key + '=([^&]*)'))
  return (match && match[1] && decodeURIComponent(match[1])) || ''
}

const toCamelCase = (obj) => {
  if (Array.isArray(obj)) {
    return obj.map(v => toCamelCase(v))
  } else if (obj !== null && obj.constructor === Object) {
    return Object.keys(obj).reduce(
      (result, key) => ({
        ...result,
        [camelCase(key)]: toCamelCase(obj[key])
      }),
      {}
    )
  }
  return obj
}

export {
  getQueryVariable,
  toCamelCase
}
