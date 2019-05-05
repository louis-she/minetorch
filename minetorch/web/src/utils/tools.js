/**
 * [获取url参数]
 * @param key
 * @param href
 */
const getQueryVariable = (key, href) => {
  let match = decodeURIComponent(href || window.location.href).match(new RegExp('[?&]' + key + '=([^&]*)'))
  return (match && match[1] && decodeURIComponent(match[1])) || ''
}

export {
  getQueryVariable
}
